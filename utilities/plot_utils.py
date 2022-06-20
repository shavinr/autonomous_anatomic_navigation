import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from simulus.utilities import get_data_folder
import seaborn as sns


mpl.style.use("seaborn")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Utopia"] + plt.rcParams["font.serif"],
    "font.size": 18,
    "axes.labelpad": 10,
    "xtick.major.pad": 0.5,
    "axes.labelsize": 'x-large',
    "axes.titlesize": 'xx-large',
    "xtick.labelsize": 'x-large',
    "ytick.labelsize": 'x-large',
    "legend.fontsize": 'x-large',
    "lines.linewidth": 2.0})


def generalized_line_plotter(rel_data_filepath, title, xlabel, ylabel, save_dir):
    # Note that data_filepath must be in the form '---.csv'
    data = pd.read_csv(rel_data_filepath, index_col=1)
    results_df = pd.DataFrame(data, columns=["Value"])

    plt.figure()
    plt.plot(results_df, color="#900C3F")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    generalized_fig_saver(save_dir, title+".pdf")
    plt.show()


def generalized_scatter_plotter(rel_data_filepath, title, xlabel, ylabel, save_dir, zlabel=None):
    # Note that data_filepath must be in the form '---.csv'
    data = pd.read_csv(rel_data_filepath, header=1)
    results_df = pd.DataFrame(data)
    results_df.rename(columns={'t': str(xlabel), 'l': str(ylabel)}, errors="raise", inplace=True)

    if zlabel is not None:
        results_df[zlabel] = results_df.iloc[:, 0].values / results_df.max(axis=0)[0]
        scatplot = sns.scatterplot(data=results_df, x=xlabel, y=ylabel, hue=results_df[zlabel], size=results_df[zlabel], sizes=(100, 1500))
    else:
        scatplot = sns.scatterplot(data=results_df, x=xlabel, y=ylabel)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    generalized_fig_saver(save_dir, title+".pdf")
    plt.show()


def generalized_fig_saver(save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight", format="pdf")


if __name__ == "__main__":
    filepath_ep_len_mean = get_data_folder()+"/../simulus/training_results/final_ep_len_mean.csv"
    filepath_ep_rew_mean = get_data_folder()+"/../simulus/training_results/final_ep_rew_mean.csv"
    filepath_eval_len_mean = get_data_folder()+"/../simulus/training_results/final_eval_mean_ep_length.csv"
    filepath_eval_rew_mean = get_data_folder()+"/../simulus/training_results/final_eval_mean_reward.csv"
    filepath_single_env = get_data_folder()+"/../simulus/training_results/case_id_2_seed_5_training=True.monitor.csv"
    save_dir = get_data_folder()+"/../../../figures"

    generalized_line_plotter(filepath_ep_len_mean, "Mean training episode length", "Timestep", "Episode length",
                             save_dir)
    generalized_line_plotter(filepath_ep_rew_mean, "Mean training episode reward", "Timestep", "Episode reward",
                             save_dir)
    generalized_line_plotter(filepath_eval_len_mean, "Mean evaluation episode length", "Timestep", "Episode length",
                             save_dir)
    generalized_line_plotter(filepath_eval_rew_mean, "Mean evaluation episode reward", "Timestep", "Episode reward",
                             save_dir)
    generalized_scatter_plotter(filepath_single_env, "Single environment episode length and reward", "Time [s]",
                                "Episode length [timesteps]", save_dir, zlabel="Relative episode reward")