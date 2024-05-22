import argparse
import os
import json

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str)

    args = parser.parse_args()
    results_path = args.path

    if not os.path.isdir(results_path):
        raise ValueError(f"Path {results_path} is not a directory!")

    ds = []
    replicate_dirs = [os.path.join(results_path, replicate_dir) for replicate_dir in os.listdir(results_path)
                      if os.path.isdir(os.path.join(results_path, replicate_dir))]
    print("Replicate dirs", replicate_dirs)
    for replicate_d in sorted(replicate_dirs):
        if os.path.isfile(os.path.join(replicate_d, "out.json")):
            log_path = os.path.join(replicate_d, "out.json")
        else:
            print(replicate_d, "has no out.json! Skipping...")

        with open(log_path) as f:
            d = json.load(f)
        ds.append(d)

    combos = set()
    # First preprocess each replicate (mostly need to average across MAMAML IDs)
    for expt in ds[0]:
        combos.add((expt['game'], expt['p1'], expt['p2']))

    ds_processed = []
    for d in ds:
        d_processed = {}
        for combo in combos:
            rews_0 = []
            rews_1 = []
            for expt in d:
                if expt['game'] == combo[0] and expt['p1'] == combo[1] and expt['p2'] == combo[2]:
                    rews_0.append(expt['rew_0'])
                    rews_1.append(expt['rew_1'])
            d_processed[combo] = (np.mean(rews_0), np.mean(rews_1))
        ds_processed.append(d_processed)

    # Now average across replications
    combo_values = []
    for combo in combos:
        rews_0 = []
        rews_1 = []
        for d_processed in ds_processed:
            rews_0.append(d_processed[combo][0])
            rews_1.append(d_processed[combo][1])
        print(combo, np.mean(rews_0), np.mean(rews_1))
