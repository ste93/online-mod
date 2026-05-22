import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import utils
import warnings

from pprint import pprint


def read_cliff_map_data(cliff_map_file):
    if not os.path.exists(cliff_map_file):
        return None
    
    data = pd.read_csv(cliff_map_file, header=None)
    data.columns = ["x", "y", "velocity", "motion_angle",
                    "cov1", "cov2", "cov3", "cov4", "weight", "motion_ratio"]

    return data.to_numpy()

def plot_cliff_map_with_weight(cliff_map_data, dataset="ATC"):
    max_index_list = []
    
    location = cliff_map_data[0, :2]
    weight = cliff_map_data[0, 8]
    orientation = cliff_map_data[:, 3]
    orientation = np.mod(orientation, 2 * np.pi)
    
    max_weight_index = 0

    for i in range(1, len(cliff_map_data)):
        tmp_location = cliff_map_data[i, :2]
        if (tmp_location == location).all():
            tmp_weight = cliff_map_data[i, 8]
            if tmp_weight > weight:
                max_weight_index = i
                weight = tmp_weight
        else:
            max_index_list.append(max_weight_index)
            location = cliff_map_data[i, :2]
            weight = cliff_map_data[i, 8]
            max_weight_index = i

    max_index_list.append(max_weight_index)

    (u, v) = utils.pol2cart(cliff_map_data[:, 2], orientation)
    weight = cliff_map_data[:, 8]

    colors = orientation  * 180 / np.pi
    colors = np.append(colors, [0, 360])
    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.hsv

    for i in range(len(cliff_map_data)):
        if dataset == "ATC":
            plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=weight[i], cmap="hsv",angles='xy', scale_units='xy', scale=1, width=0.003)
        elif dataset == "MAPF":
            plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=weight[i], cmap="hsv",angles='xy', scale_units='xy', scale=0.5)
        elif dataset == "MADAMA":
            plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=weight[i], cmap="hsv",angles='xy', scale_units='xy', scale=1, width=0.003)
    
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    cbar = plt.colorbar(sm, shrink = 0.5, ticks=[0, 90, 180, 270, 360], fraction=0.05)
    
    if dataset == "ATC":
        cbar.ax.tick_params(labelsize=10)
        plt.text(100, -17,"Orientation [deg]", rotation='vertical')
    elif dataset == "MAPF":
        cbar.ax.tick_params(labelsize=15)
        plt.text(121, 29,"Orientation [deg]", rotation='vertical', fontsize=15)
    elif dataset == "MADAMA":
        cbar.ax.tick_params(labelsize=10)
        

def plot_cliff_map_atc(cliff_file_name, output_fig_name):
    cliff_map_data = read_cliff_map_data(cliff_file_name)
    plt.clf()
    plt.close('all')
    plt.figure(figsize=(10, 6))
    plt.subplot(111, facecolor='white')
    img = plt.imread("maps/localization_grid_white.jpg")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255, extent=[-60, 80, -40, 20])
    plot_cliff_map_with_weight(cliff_map_data, dataset="ATC")
    plt.savefig(output_fig_name)


def plot_cliff_map_mapf(cliff_file_name, output_fig_name):
    cliff_map_data = read_cliff_map_data(cliff_file_name)
    plt.clf()
    plt.close('all')
    plt.figure(figsize=(6, 6), dpi=100)
    plt.subplot(111, facecolor='white')
    img = plt.imread(f"maps/den_small.png")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255, extent=[0, 100, 0, 100])
    plot_cliff_map_with_weight(cliff_map_data, dataset="MAPF")
    plt.savefig(output_fig_name)


def plot_cliff_map_madama(cliff_file_name, output_fig_name):
    cliff_map_data = read_cliff_map_data(cliff_file_name)
    plt.clf()
    plt.close('all')
    plt.figure(figsize=(8, 8), dpi=100)
    plt.subplot(111, facecolor='white')
    img = plt.imread("maps/madama.png")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255, extent=[-40, 25, -35, 25])
    plot_cliff_map_with_weight(cliff_map_data, dataset="MADAMA")
    plt.savefig(output_fig_name)

    
if __name__ == "__main__":
    
    ############# For plotting ATC cliffmaps #############
    # os.makedirs(f"cliffmaps/atc/{exp_type}/figs", exist_ok=True)
    # for hour in range(9, 21):
    #     plot_cliff_map_atc(f"cliffmaps/atc/{exp_type}/atc_1024_{hour}_{hour + 1}.csv", f"cliffmaps/atc/{exp_type}/figs/atc_1024_{hour}_{hour + 1}.png")
    ######################################################
        
        
    ############# For plotting MAPF cliffmaps #############
    exp_type = "interval" # "online" or "all" or "interval"
    os.makedirs(f"cliffmaps/mapf/{exp_type}/figs", exist_ok=True)
    for version in ["initial", "update"]:
        for batch in range(1, 11):
            plot_cliff_map_mapf(f"cliffmaps/mapf/{exp_type}/{version}_split_b{batch}.csv", f"cliffmaps/mapf/{exp_type}/figs/{version}_split_b{batch}.png")
    ######################################################

    ############# For plotting MADAMA cliffmaps #############
    # os.makedirs(f"cliffmaps/madama/{exp_type}/figs", exist_ok=True)
    # for month in ["2024_06", "2024_07", "2024_10", "2024_11"]:
    #     plot_cliff_map_madama(f"cliffmaps/madama/{exp_type}/madama_{month}.csv", f"cliffmaps/madama/{exp_type}/figs/madama_{month}.png")
    ######################################################
