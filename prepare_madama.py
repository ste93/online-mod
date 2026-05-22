#!/usr/bin/env python3

import argparse
import calendar
import csv
import math
import random
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median


MADAMA_TIME_FORMAT = "%m/%d/%Y %H:%M:%S"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "dataset" / "madama"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "split_data_random"
DEFAULT_REFERENCE_DIR = SCRIPT_DIR / "dataset" / "atc" / "split_data_random"
FALLBACK_MIN_TRACK_LENGTH = 29
FALLBACK_MAX_TRACK_LENGTH = 78
FALLBACK_TARGET_TOTAL_ROWS = 65000


@dataclass
class Track:
    track_id: int
    x: float
    y: float


@dataclass
class ReferenceStats:
    min_track_length: int
    max_track_length: int
    target_total_rows: int
    source: str


@dataclass
class TrackSegment:
    output_file: str
    offset: int
    length: int


@dataclass
class SplitResult:
    train_file: Path
    test_file: Path
    train_count: int = None
    test_count: int = None
    skipped: bool = False
    raw_motion_rows: int = 0
    eligible_tracks: int = 0
    selected_tracks: int = 0
    selected_rows: int = 0


class TrackAssigner:
    def __init__(self, max_speed: float, max_gap: float) -> None:
        self.max_speed = max_speed
        self.max_gap = max_gap
        self.next_track_id = 1
        self.previous_time = None
        self.previous_tracks = []

    def process_frame(self, time_s, points):
        if self.previous_time is None:
            self._start_new_tracks(time_s, points)
            return []

        dt = time_s - self.previous_time
        if dt <= 0 or dt > self.max_gap:
            self._start_new_tracks(time_s, points)
            return []

        max_distance = self.max_speed * dt
        candidates = []
        for track_idx, track in enumerate(self.previous_tracks):
            for point_idx, point in enumerate(points):
                distance = math.dist((track.x, track.y), point)
                if distance <= max_distance:
                    candidates.append((distance, track_idx, point_idx))

        candidates.sort(key=lambda item: item[0])
        matched_tracks = set()
        matched_points = set()
        point_track_ids = {}
        rows = []

        for distance, track_idx, point_idx in candidates:
            if track_idx in matched_tracks or point_idx in matched_points:
                continue

            track = self.previous_tracks[track_idx]
            x, y = points[point_idx]
            velocity = distance / dt
            motion_angle = math.atan2(y - track.y, x - track.x) % (2 * math.pi)
            rows.append((time_s, track.track_id, x, y, velocity, motion_angle))
            matched_tracks.add(track_idx)
            matched_points.add(point_idx)
            point_track_ids[point_idx] = track.track_id

        current_tracks = []
        for point_idx, (x, y) in enumerate(points):
            track_id = point_track_ids.get(point_idx)
            if track_id is None:
                track_id = self._new_track_id()
            current_tracks.append(Track(track_id, x, y))

        self.previous_time = time_s
        self.previous_tracks = current_tracks
        return rows

    def _start_new_tracks(self, time_s, points):
        self.previous_time = time_s
        self.previous_tracks = [Track(self._new_track_id(), x, y) for x, y in points]

    def _new_track_id(self):
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw MADAMA detection files into ATC-style train/test motion CSVs. "
            "By default, paths are resolved next to this script, so it can be run from any directory."
        )
    )
    parser.add_argument(
        "--input-file",
        action="append",
        type=Path,
        default=[],
        help="Raw MADAMA CSV file to process. Can be passed multiple times. Overrides --input-dir/--raw-glob."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing raw MADAMA CSV files. Default: {DEFAULT_INPUT_DIR}"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where split CSVs are written. Default: {DEFAULT_OUTPUT_DIR}"
    )
    parser.add_argument(
        "--raw-glob",
        default="combined_detections_*.csv",
        help="Glob used inside --input-dir when --input-file is not set."
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Target fraction of selected track rows to write to *_test.csv."
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help=f"ATC split directory used for auto sizing. Default: {DEFAULT_REFERENCE_DIR}"
    )
    parser.add_argument(
        "--min-track-length",
        default="auto",
        help=(
            "Minimum rows in a reconstructed MADAMA track. "
            "'auto' uses the ATC median, currently about 29."
        )
    )
    parser.add_argument(
        "--max-track-length",
        default="auto",
        help=(
            "Maximum rows kept from each reconstructed MADAMA track. "
            "Use 'all' or 'none' to disable. 'auto' uses the ATC 90th percentile."
        )
    )
    parser.add_argument(
        "--target-total-rows",
        default="auto",
        help=(
            "Approximate total train+test rows per output pair. "
            "Use 'all' or 'none' to keep every eligible track. 'auto' uses the ATC median file size."
        )
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for deterministic splits.")
    parser.add_argument(
        "--max-speed",
        type=float,
        default=3.0,
        help="Maximum plausible speed in m/s for linking detections across frames."
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=2.0,
        help="Maximum time gap in seconds for linking detections across frames."
    )
    parser.add_argument("--max-raw-rows", type=int, default=None, help="Optional raw-row limit for smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing split files.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip raw files whose split outputs already exist.")

    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args):
    if not 0 <= args.test_ratio <= 1:
        raise ValueError("--test-ratio must be between 0 and 1")
    if args.max_speed <= 0:
        raise ValueError("--max-speed must be positive")
    if args.max_gap <= 0:
        raise ValueError("--max-gap must be positive")
    if args.max_raw_rows is not None and args.max_raw_rows <= 0:
        raise ValueError("--max-raw-rows must be positive")
    if args.overwrite and args.skip_existing:
        raise ValueError("--overwrite and --skip-existing cannot be used together")
    validate_auto_int(args.min_track_length, "--min-track-length", allow_none=False)
    validate_auto_int(args.max_track_length, "--max-track-length", allow_none=True)
    validate_auto_int(args.target_total_rows, "--target-total-rows", allow_none=True)


def validate_auto_int(value, name, allow_none):
    lowered = str(value).lower()
    if lowered == "auto":
        return
    if allow_none and lowered in {"all", "none"}:
        return
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be 'auto' or a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")


def parse_time(value):
    dt = datetime.strptime(value, MADAMA_TIME_FORMAT)
    return float(calendar.timegm(dt.timetuple()))


def parse_points(row):
    if len(row) < 2:
        return []

    try:
        detection_count = int(float(row[1]))
    except ValueError:
        return []

    coords = row[2:]
    point_count = min(detection_count, len(coords) // 2)
    points = []
    for point_idx in range(point_count):
        try:
            x = float(coords[point_idx * 2])
            y = float(coords[point_idx * 2 + 1])
        except ValueError:
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    return points


def iter_frames(raw_file, max_raw_rows):
    current_time = None
    current_points = []

    with raw_file.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row_count, row in enumerate(reader, start=1):
            if max_raw_rows is not None and row_count > max_raw_rows:
                break
            if not row:
                continue

            frame_time = parse_time(row[0])
            frame_points = parse_points(row)
            if current_time is None:
                current_time = frame_time
                current_points = frame_points
            elif frame_time == current_time:
                current_points = frame_points
            else:
                yield current_time, current_points
                current_time = frame_time
                current_points = frame_points

    if current_time is not None:
        yield current_time, current_points


def output_stem(raw_file):
    match = re.search(r"combined_detections_(\d{2})-\d{2}-(\d{4})", raw_file.name)
    if match is None:
        return raw_file.stem

    month, year = match.groups()
    return f"madama_{year}_{month}"


def format_motion_row(row):
    time_s, person_id, x, y, velocity, motion_angle = row
    return [
        round(time_s, 3),
        person_id,
        round(x, 5),
        round(y, 5),
        round(velocity, 5),
        round(motion_angle, 5),
    ]


def percentile(values, quantile):
    if not values:
        raise ValueError("Cannot compute percentile of an empty list")

    ordered = sorted(values)
    index = int((len(ordered) - 1) * quantile)
    return ordered[index]


def atc_reference_stats(reference_dir):
    reference_dir = reference_dir.expanduser().resolve()
    if not reference_dir.is_dir():
        return ReferenceStats(
            FALLBACK_MIN_TRACK_LENGTH,
            FALLBACK_MAX_TRACK_LENGTH,
            FALLBACK_TARGET_TOTAL_ROWS,
            f"fallback; ATC reference dir not found: {reference_dir}",
        )

    track_lengths = []
    paired_row_counts = {}
    for csv_file in sorted(reference_dir.glob("*.csv")):
        if csv_file.name.endswith("_train.csv"):
            pair_key = csv_file.name[:-len("_train.csv")]
        elif csv_file.name.endswith("_test.csv"):
            pair_key = csv_file.name[:-len("_test.csv")]
        else:
            continue

        row_count = 0
        track_counts = Counter()
        with csv_file.open(newline="") as file_obj:
            for row in csv.reader(file_obj):
                if not row:
                    continue
                row_count += 1
                track_counts[row[1]] += 1

        paired_row_counts[pair_key] = paired_row_counts.get(pair_key, 0) + row_count
        track_lengths.extend(track_counts.values())

    if not track_lengths or not paired_row_counts:
        return ReferenceStats(
            FALLBACK_MIN_TRACK_LENGTH,
            FALLBACK_MAX_TRACK_LENGTH,
            FALLBACK_TARGET_TOTAL_ROWS,
            f"fallback; no ATC split rows found in {reference_dir}",
        )

    return ReferenceStats(
        max(1, round(median(track_lengths))),
        max(1, percentile(track_lengths, 0.90)),
        max(1, round(median(paired_row_counts.values()))),
        str(reference_dir),
    )


def resolve_auto_int(value, auto_value, allow_none):
    lowered = str(value).lower()
    if lowered == "auto":
        return auto_value
    if allow_none and lowered in {"all", "none"}:
        return None
    return int(value)


def resolve_reference_settings(args):
    stats = atc_reference_stats(args.reference_dir)
    min_track_length = resolve_auto_int(args.min_track_length, stats.min_track_length, allow_none=False)
    max_track_length = resolve_auto_int(args.max_track_length, stats.max_track_length, allow_none=True)
    target_total_rows = resolve_auto_int(args.target_total_rows, stats.target_total_rows, allow_none=True)

    if max_track_length is not None and max_track_length < min_track_length:
        raise ValueError("--max-track-length must be greater than or equal to --min-track-length")

    print(
        "Sizing MADAMA from ATC reference "
        f"({stats.source}): min_track_length={min_track_length}, "
        f"max_track_length={max_track_length if max_track_length is not None else 'all'}, "
        f"target_total_rows={target_total_rows if target_total_rows is not None else 'all'}"
    )

    return min_track_length, max_track_length, target_total_rows


def write_motion_temp(raw_file, temp_file, args):
    assigner = TrackAssigner(max_speed=args.max_speed, max_gap=args.max_gap)
    track_lengths = Counter()
    raw_motion_rows = 0

    with temp_file.open("w", newline="") as temp_csv:
        writer = csv.writer(temp_csv)
        for frame_time, points in iter_frames(raw_file, args.max_raw_rows):
            for motion_row in assigner.process_frame(frame_time, points):
                writer.writerow(format_motion_row(motion_row))
                track_lengths[motion_row[1]] += 1
                raw_motion_rows += 1

    return track_lengths, raw_motion_rows


def effective_track_lengths(track_lengths, min_track_length, max_track_length):
    effective_lengths = {}
    for track_id, track_length in track_lengths.items():
        if track_length < min_track_length:
            continue
        if max_track_length is None:
            effective_lengths[track_id] = track_length
        else:
            effective_lengths[track_id] = min(track_length, max_track_length)
    return effective_lengths


def select_tracks(effective_lengths, target_total_rows, rng):
    track_ids = list(effective_lengths)
    rng.shuffle(track_ids)

    if target_total_rows is None:
        return set(track_ids)

    selected = set()
    selected_rows = 0
    for track_id in track_ids:
        if selected_rows >= target_total_rows:
            break
        selected.add(track_id)
        selected_rows += effective_lengths[track_id]

    return selected


def split_tracks(selected_tracks, effective_lengths, test_ratio, rng):
    track_ids = list(selected_tracks)
    rng.shuffle(track_ids)
    target_test_rows = round(sum(effective_lengths[track_id] for track_id in track_ids) * test_ratio)
    test_tracks = set()
    test_rows = 0

    for track_id in track_ids:
        if test_rows >= target_test_rows:
            break
        test_tracks.add(track_id)
        test_rows += effective_lengths[track_id]

    return test_tracks


def select_track_segments(track_lengths, effective_lengths, selected_tracks, test_tracks, rng):
    segments = {}
    for track_id in selected_tracks:
        segment_length = effective_lengths[track_id]
        full_length = track_lengths[track_id]
        max_offset = full_length - segment_length
        offset = rng.randint(0, max_offset) if max_offset > 0 else 0
        output_file = "test" if track_id in test_tracks else "train"
        segments[track_id] = TrackSegment(output_file, offset, segment_length)
    return segments


def write_selected_segments(temp_file, train_file, test_file, segments):
    seen_counts = Counter()
    train_count = 0
    test_count = 0

    with temp_file.open(newline="") as temp_csv, train_file.open("w", newline="") as train_csv, test_file.open("w", newline="") as test_csv:
        reader = csv.reader(temp_csv)
        train_writer = csv.writer(train_csv)
        test_writer = csv.writer(test_csv)

        for row in reader:
            if not row:
                continue

            track_id = int(row[1])
            segment = segments.get(track_id)
            if segment is None:
                continue

            row_index = seen_counts[track_id]
            seen_counts[track_id] += 1
            if row_index < segment.offset or row_index >= segment.offset + segment.length:
                continue

            if segment.output_file == "test":
                test_writer.writerow(row)
                test_count += 1
            else:
                train_writer.writerow(row)
                train_count += 1

    return train_count, test_count


def split_raw_file(raw_file, output_dir, args, reference_settings):
    stem = output_stem(raw_file)
    train_file = output_dir / f"{stem}_train.csv"
    test_file = output_dir / f"{stem}_test.csv"
    if not args.overwrite and (train_file.exists() or test_file.exists()):
        if args.skip_existing:
            return SplitResult(train_file, test_file, skipped=True)
        raise FileExistsError(f"{train_file} or {test_file} already exists. Use --overwrite to replace it.")

    file_seed = args.seed + zlib.crc32(raw_file.name.encode("utf-8"))
    rng = random.Random(file_seed)
    min_track_length, max_track_length, target_total_rows = reference_settings
    temp_file = output_dir / f".{stem}_motion.tmp.csv"

    try:
        track_lengths, raw_motion_rows = write_motion_temp(raw_file, temp_file, args)
        effective_lengths = effective_track_lengths(track_lengths, min_track_length, max_track_length)
        selected_tracks = select_tracks(effective_lengths, target_total_rows, rng)
        test_tracks = split_tracks(selected_tracks, effective_lengths, args.test_ratio, rng)
        segments = select_track_segments(track_lengths, effective_lengths, selected_tracks, test_tracks, rng)
        train_count, test_count = write_selected_segments(temp_file, train_file, test_file, segments)

        return SplitResult(
            train_file=train_file,
            test_file=test_file,
            train_count=train_count,
            test_count=test_count,
            raw_motion_rows=raw_motion_rows,
            eligible_tracks=len(effective_lengths),
            selected_tracks=len(selected_tracks),
            selected_rows=train_count + test_count,
        )
    finally:
        temp_file.unlink(missing_ok=True)


def find_raw_files(args):
    if args.input_file:
        raw_files = [raw_file.expanduser().resolve() for raw_file in args.input_file]
        missing_files = [raw_file for raw_file in raw_files if not raw_file.is_file()]
        if missing_files:
            missing_list = ", ".join(str(raw_file) for raw_file in missing_files)
            raise FileNotFoundError(f"Raw MADAMA file(s) not found: {missing_list}")
        return sorted(raw_files)

    input_dir = args.input_dir.expanduser().resolve()
    raw_files = sorted(input_dir.glob(args.raw_glob))
    if not raw_files:
        raise FileNotFoundError(f"No MADAMA raw files found in {input_dir} matching {args.raw_glob}")
    return raw_files


def main():
    args = get_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_settings = resolve_reference_settings(args)
    raw_files = find_raw_files(args)
    for raw_file in raw_files:
        result = split_raw_file(raw_file, output_dir, args, reference_settings)
        if result.skipped:
            print(f"{raw_file.name}: skipped existing outputs {result.train_file} and {result.test_file}")
        else:
            print(
                f"{raw_file.name}: reconstructed {result.raw_motion_rows} motion rows, "
                f"kept {result.selected_rows} rows from {result.selected_tracks}/"
                f"{result.eligible_tracks} eligible tracks; wrote {result.train_count} train rows "
                f"to {result.train_file} and {result.test_count} test rows to {result.test_file}"
            )


if __name__ == "__main__":
    main()
