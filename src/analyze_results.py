import argparse
import os
import json

import numpy as np
from matplotlib import pyplot as plt


def get_stat_names(d: list):
    return list(d[0].keys())


def average_stat(ds, stat_name, symmetric: bool = False):
    ys = []
    if not symmetric:
        for i, d in enumerate(ds):
            ys.append(np.array([ep_data.get(stat_name, 0) for ep_data in d]))
    else:
        for i, d_ in enumerate(ds):
            ys.append(np.array([ep_data.get(stat_name, 0) for ep_data in d_]))
    ys = np.stack(ys, axis=0)
    return ys.mean(axis=0), ys.std(axis=0) / ys.shape[0]


def _plot_line(xs, ys, sem=None, N: int = 1):
    plt.plot(xs, np.convolve(ys, np.ones(N) / N, mode='same'))
    if sem is not None:
        plt.fill_between(xs, ys - sem, ys + sem, alpha=0.5)


def plot_stat(d: list or dict, stat_name: str or list, N: int = 1, symmetric: bool = False, save: bool = False):
    """
    Plot experimental results from a single stat.
    :param d: Either a list of dictionaries from multiple experiments or a single one.
    :param stat_name: The key in the dictionary to plot.
    :param N: The window size for smoothing.
    :param symmetric: Whether to combine symmetric agents (only applicable for multiple expts)
    :param save: Whether to save the plot.
    """
    if isinstance(d[0], dict):
        ys = np.array([ep_data.get(stat_name, 0) for ep_data in d])
        xs = range(ys.size)
        _plot_line(xs, ys, N=N)
        print(f"End {stat_name}: {ys[-1]:.2f}")
    elif isinstance(d[0], list):
        y_mean, y_sem = average_stat(d, stat_name, symmetric=symmetric)
        xs = range(y_mean.size)
        _plot_line(xs, y_mean, sem=y_sem, N=N)
        print(f"Mean {stat_name}: {y_mean[-1]:.2f} ± {y_sem[-1]:.2f}")
    else:
        raise ValueError

    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel(stat_name.replace('_', ' ').title())
    # plt.title(stat_name)
    if save:
        plt.savefig(f"figures/{stat_name}.pdf")
    plt.show()


def get_last_log_path(dir_path: str):
    if not os.path.isdir(dir_path):
        print(dir_path, "not a directory")
        return
    results = [f[4:-5] for f in os.listdir(dir_path) if f.startswith('out_')]
    if len(results) == 0:
        print("No results found for", dir_path)
        return
    last_result_num = str(sorted([int(r) for r in results])[-1])
    return os.path.join(dir_path, f'out_{last_result_num}.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('-sym', '--symmetric', action='store_true')
    parser.add_argument('-t', '--truncate', type=int, default=None)
    parser.add_argument("-s", "--save", action='store_true')

    args = parser.parse_args()
    symmetric = args.symmetric
    results_path = args.path
    truncate_len = args.truncate
    save = args.save

    if os.path.isfile(results_path):
        with open(results_path) as f:
            d = json.load(f)
        stat_names = get_stat_names(d)
        for stat_name in stat_names:
            plot_stat(d, stat_name, save=save)

    elif os.path.isdir(results_path):
        ds = []
        replicate_dirs = [replicate_dir for replicate_dir in os.listdir(results_path)
                          if os.path.isdir(os.path.join(results_path, replicate_dir))]
        print("Replicate dirs", replicate_dirs)
        for replicate_d in sorted(replicate_dirs):
            last_log_path = get_last_log_path(os.path.join(results_path, replicate_d))
            if last_log_path is None:
                continue
            print("Logs", last_log_path)
            with open(last_log_path) as f:
                d = json.load(f)
                ds.append(d)

        min_len = min([len(d_) for d_ in ds])
        if truncate_len is not None and truncate_len > min_len:
            ds_temp = ds
            ds = []
            for d_ in ds_temp:
                if len(d_) >= truncate_len:
                    ds.append(d_[:truncate_len])
                else:
                    print(f"Skipping {len(d_)} < {truncate_len}")
        elif truncate_len is not None:
            print("Truncating to length:", truncate_len)
            ds = [d_[:truncate_len] for d_ in ds]
        else:
            print("Truncating to min length:", min_len)
            ds = [d_[:min_len] for d_ in ds]

        stat_names = get_stat_names(ds[0])
        for stat_name in stat_names:
            plot_stat(ds, stat_name, symmetric=symmetric, save=save)
