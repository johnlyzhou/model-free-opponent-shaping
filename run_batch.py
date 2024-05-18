import argparse
import json
import os
import pathlib
from time import sleep
import multiprocessing as mp

import torch

from src.coin_game.main_mfos_coin_game import main_mfos_coin_game
from src.coin_game.main_mfos_self_coin_game import main_mfos_self_coin_game


CUDA_LIST = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="test")
    parser.add_argument("-g", "--game", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default=None)
    parser.add_argument("-r", "--replicate", type=int, default=1)
    parser.add_argument("-dd", "--results_dir", type=str, default="results")
    args = parser.parse_args()

    expt_name = args.name
    results_dir = args.results_dir
    game_name = args.game.lower()
    if game_name == "mfos_self_coin_game":
        train_func = main_mfos_self_coin_game
    elif game_name == "main_mfos_coin_game":
        train_func = main_mfos_coin_game
    else:
        raise NotImplementedError(f"Game {game_name} not implemented (yet).")

    if args.device == "all":
        device_list = CUDA_LIST
        print("Using all available devices:", device_list)
    elif args.device is not None:
        device_list = [torch.device(args.device)]
        print("Using:", device_list[0])
    else:
        device_list = [torch.device("cpu")]
        print("No device specified, using CPU")

    processes = [None for _ in range(len(device_list))]

    for i in range(args.replicate):
        if args.replicate > 1:
            save_dir = f"{expt_name}_{i}"
        else:
            save_dir = expt_name
        expt_dir = os.path.join(results_dir, save_dir)
        if not os.path.isdir(expt_dir):
            pathlib.Path(expt_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(expt_dir, "commandline_args.txt"), "w") as f:
                json.dump(args.__dict__, f, indent=2)

        while True:
            try:
                device_idx = processes.index(None)
                device = device_list[device_idx]
                break
            except ValueError:
                for i, p in enumerate(processes):
                    if not p.is_alive():
                        processes[i] = None
                sleep(1)

        process_args = (expt_dir, device)

        processes[device_idx] = mp.Process(target=train_func, args=process_args)
        processes[device_idx].start()
