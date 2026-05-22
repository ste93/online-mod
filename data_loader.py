import pandas as pd
import numpy as np

import utils

MOTION_DATASETS = {"ATC", "MAGNI", "MAPF", "MADAMA"}


class DataLoader():

    def __init__(
        self,
        config_file: str,
        raw_data_file: str,
    ) -> None:
        self.config_params = utils.read_config_file(config_file)
        self.raw_data_file = raw_data_file
        self.map_file = self.config_params['map_file']
        self.raw_dataset = self.config_params['raw_dataset']
        self._load_dataset()
    
    def _load_dataset(self):
        if self.raw_dataset in MOTION_DATASETS:
            self._load_motion_dataset()
        else:
            raise ValueError(f"Unsupported raw_dataset: {self.raw_dataset}")

    def _load_motion_dataset(self):
        data = pd.read_csv(self.raw_data_file, header=None)
        if data.shape[1] != 6:
            raise ValueError(
                f"{self.raw_dataset} data must have 6 columns: "
                "time, person_id, x, y, velocity, motion_angle. "
                "For raw MADAMA detection files, run prepare_madama.py first."
            )

        data.columns = ["time", "person_id", "x", "y", "velocity", "motion_angle"]
        data['motion_angle'] = np.mod(data['motion_angle'], 2 * np.pi)
        self.data = data[['time', 'x', 'y', 'velocity', 'motion_angle']]
            

    def in_fov(self, x, y, robot_pos, facing_angle, fov_angle, fov_radius):
        rel_pos = np.array([x, y]) - robot_pos
        distance = np.linalg.norm(rel_pos)
        angle = np.arctan2(rel_pos[1], rel_pos[0])
        
        angle_diff = (angle - facing_angle + np.pi) % (2 * np.pi) - np.pi
        
        return (distance <= fov_radius) and (abs(angle_diff) <= fov_angle / 2)
    
    def in_region(self, x, y, obs_x, obs_y, delta_x, delta_y):
        return (x >= obs_x and x <= obs_x + delta_x and y >= obs_y and y <= obs_y + delta_y)

    def get_observed_traj(self, robot_position, robot_facing_angle, fov_angle, fov_radius, observe_start_time, observe_period):
        # Filter data based on observation time window
        time_filtered_df = self.data[(self.data['time'] >= observe_start_time) & (self.data['time'] < observe_start_time + observe_period)].copy()
        
        # Filter data within the robot's field of view
        time_filtered_df.loc[:, 'in_fov'] = time_filtered_df.apply(
            lambda row: self.in_fov(row['x'], row['y'], robot_position, robot_facing_angle, fov_angle, fov_radius), axis=1
        )
        
        df_in_fov = time_filtered_df[time_filtered_df['in_fov']].drop(columns=['in_fov'])
        
        return df_in_fov
        

    def get_observed_traj_region(self, obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period):
        # Filter data based on observation time window
        time_filtered_df = self.data[(self.data['time'] >= observe_start_time) & (self.data['time'] < observe_start_time + observe_period)].copy()
        # Filter data within the robot's field of view
        time_filtered_df.loc[:, 'in_fov'] = time_filtered_df.apply(
            lambda row: self.in_region(row['x'], row['y'], obs_x, obs_y, delta_x, delta_y), axis=1
        )
        
        df_in_fov = time_filtered_df[time_filtered_df['in_fov']].drop(columns=['in_fov'])
        
        return df_in_fov

    def get_observed_traj_region_all_time(self, obs_x, obs_y, delta_x, delta_y):
        # Filter data based on observation time window
        time_filtered_df = self.data.copy()
        
        # Filter data within the robot's field of view
        time_filtered_df.loc[:, 'in_fov'] = time_filtered_df.apply(
            lambda row: self.in_region(row['x'], row['y'], obs_x, obs_y, delta_x, delta_y), axis=1
        )
        
        df_in_fov = time_filtered_df[time_filtered_df['in_fov']].drop(columns=['in_fov'])
        
        return df_in_fov
    
    def get_observed_traj_all_area_all_time(self):
        df_in_fov = self.data.copy()
        
        return df_in_fov

    def get_observed_traj_all_area(self, obs_x, obs_y, delta_x, delta_y, observe_start_time, observe_period):
        # Filter data based on observation time window
        time_filtered_df = self.data[(self.data['time'] >= observe_start_time) & (self.data['time'] < observe_start_time + observe_period)].copy()
        
        df_in_fov = time_filtered_df
        
        return df_in_fov
