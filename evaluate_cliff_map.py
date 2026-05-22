#!/usr/bin/env python3

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal


CLIFF_COLUMNS = 10
TEST_COLUMNS = 6
EPS = 1e-12


@dataclass
class MotionComponent:
    speed: float
    angle: float
    covariance: np.ndarray
    weight: float


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a CLiFF-map against held-out motion rows."
    )
    parser.add_argument(
        "--cliff-map",
        required=True,
        type=Path,
        help="CLiFF-map CSV to evaluate, for example cliffmaps/madama/all/madama_2024_11.csv."
    )
    parser.add_argument(
        "--test-file",
        action="append",
        type=Path,
        default=[],
        help="Held-out test CSV. Can be passed multiple times."
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Directory containing *_test.csv files. Used when --test-file is not passed."
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Maximum distance from a test point to the nearest CLiFF-map grid center."
    )
    parser.add_argument(
        "--wind-num",
        type=int,
        default=1,
        help="Number of 2*pi angle wraps to include when scoring circular likelihood."
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for quick smoke tests."
    )
    return parser.parse_args()


def wrap_angle_diff(angle_a, angle_b):
    return (angle_a - angle_b + math.pi) % (2 * math.pi) - math.pi


def polar_to_cart(speed, angle):
    return np.array([speed * math.cos(angle), speed * math.sin(angle)])


def load_cliff_map(path):
    if not path.is_file():
        raise FileNotFoundError(f"CLiFF-map not found: {path}")

    grid = {}
    with path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if not row:
                continue
            if len(row) != CLIFF_COLUMNS:
                raise ValueError(f"{path} has a row with {len(row)} columns; expected {CLIFF_COLUMNS}")

            x, y = float(row[0]), float(row[1])
            component = MotionComponent(
                speed=float(row[2]),
                angle=float(row[3]) % (2 * math.pi),
                covariance=np.array([[float(row[4]), float(row[5])], [float(row[6]), float(row[7])]]),
                weight=max(float(row[8]), 0.0),
            )
            grid.setdefault((x, y), []).append(component)

    if not grid:
        raise ValueError(f"CLiFF-map is empty: {path}")

    centers = np.array(list(grid.keys()), dtype=float)
    return grid, centers


def find_nearest_components(x, y, grid, centers, radius):
    distances = np.linalg.norm(centers - np.array([x, y]), axis=1)
    nearest_index = int(np.argmin(distances))
    if distances[nearest_index] > radius:
        return None

    center = tuple(centers[nearest_index])
    return grid[center]


def component_likelihood(speed, angle, component, wind_values):
    likelihood = 0.0
    for wind in wind_values:
        point = np.array([speed, angle + 2 * math.pi * wind])
        likelihood += multivariate_normal.pdf(
            point,
            mean=np.array([component.speed, component.angle]),
            cov=component.covariance,
            allow_singular=True,
        ) * component.weight
    return likelihood


def mixture_likelihood(speed, angle, components, wind_values):
    return sum(component_likelihood(speed, angle, component, wind_values) for component in components)


def mode_prediction(components):
    return max(components, key=lambda component: component.weight)


def mean_vector_prediction(components):
    weights = np.array([component.weight for component in components], dtype=float)
    total_weight = np.sum(weights)
    if total_weight <= 0:
        weights = np.ones(len(components), dtype=float) / len(components)
    else:
        weights = weights / total_weight

    vectors = np.array([polar_to_cart(component.speed, component.angle) for component in components])
    return np.sum(vectors * weights[:, None], axis=0)


def iter_test_rows(paths, max_rows):
    seen_rows = 0
    for path in paths:
        with path.open(newline="") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if not row:
                    continue
                if len(row) != TEST_COLUMNS:
                    raise ValueError(f"{path} has a row with {len(row)} columns; expected {TEST_COLUMNS}")
                yield path, row
                seen_rows += 1
                if max_rows is not None and seen_rows >= max_rows:
                    return


def resolve_test_files(args):
    if args.test_file:
        paths = [path.expanduser().resolve() for path in args.test_file]
    else:
        test_dir = args.test_dir
        if test_dir is None:
            test_dir = Path("dataset/madama/split_data_random")
        paths = sorted(test_dir.expanduser().resolve().glob("*_test.csv"))

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing test file(s): " + ", ".join(str(path) for path in missing))
    if not paths:
        raise FileNotFoundError("No test files found. Pass --test-file or --test-dir.")
    return paths


def evaluate(cliff_map, test_paths, radius, wind_num, max_rows):
    grid, centers = load_cliff_map(cliff_map)
    wind_values = range(-wind_num, wind_num + 1)

    total = 0
    covered = 0
    nll_sum = 0.0
    speed_abs_error = []
    angle_abs_error = []
    mode_vector_sq_error = []
    mean_vector_sq_error = []
    per_file = {}

    for path, row in iter_test_rows(test_paths, max_rows):
        total += 1
        per_file.setdefault(path.name, {"total": 0, "covered": 0})
        per_file[path.name]["total"] += 1

        x = float(row[2])
        y = float(row[3])
        speed = float(row[4])
        angle = float(row[5]) % (2 * math.pi)
        components = find_nearest_components(x, y, grid, centers, radius)
        if components is None:
            continue

        covered += 1
        per_file[path.name]["covered"] += 1

        likelihood = max(mixture_likelihood(speed, angle, components, wind_values), EPS)
        nll_sum += -math.log(likelihood)

        mode = mode_prediction(components)
        actual_vector = polar_to_cart(speed, angle)
        mode_vector = polar_to_cart(mode.speed, mode.angle)
        mean_vector = mean_vector_prediction(components)

        speed_abs_error.append(abs(mode.speed - speed))
        angle_abs_error.append(abs(wrap_angle_diff(mode.angle, angle)))
        mode_vector_sq_error.append(float(np.sum((mode_vector - actual_vector) ** 2)))
        mean_vector_sq_error.append(float(np.sum((mean_vector - actual_vector) ** 2)))

    if covered == 0:
        raise ValueError("No test rows were covered by the CLiFF-map. Check --radius and coordinate frames.")

    return {
        "total": total,
        "covered": covered,
        "coverage": covered / total if total else 0,
        "mean_nll": nll_sum / covered,
        "speed_mae": float(np.mean(speed_abs_error)),
        "angle_mae_deg": float(np.degrees(np.mean(angle_abs_error))),
        "mode_vector_rmse": math.sqrt(float(np.mean(mode_vector_sq_error))),
        "mean_vector_rmse": math.sqrt(float(np.mean(mean_vector_sq_error))),
        "per_file": per_file,
    }


def main():
    args = get_args()
    test_paths = resolve_test_files(args)
    results = evaluate(
        cliff_map=args.cliff_map.expanduser().resolve(),
        test_paths=test_paths,
        radius=args.radius,
        wind_num=args.wind_num,
        max_rows=args.max_rows,
    )

    print(f"rows: {results['covered']}/{results['total']} covered ({results['coverage']:.2%})")
    print(f"mean negative log-likelihood: {results['mean_nll']:.4f}")
    print(f"mode speed MAE: {results['speed_mae']:.4f} m/s")
    print(f"mode angle MAE: {results['angle_mae_deg']:.2f} deg")
    print(f"mode vector RMSE: {results['mode_vector_rmse']:.4f} m/s")
    print(f"mixture-mean vector RMSE: {results['mean_vector_rmse']:.4f} m/s")
    print("per-file coverage:")
    for name, item in sorted(results["per_file"].items()):
        coverage = item["covered"] / item["total"] if item["total"] else 0
        print(f"  {name}: {item['covered']}/{item['total']} ({coverage:.2%})")


if __name__ == "__main__":
    main()
