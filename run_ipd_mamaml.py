import argparse
import json
import os
from time import sleep
import multiprocessing as mp

import torch

from src.main_mfos_ppo import main

CUDA_LIST = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, required=True)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not os.path.isdir(args.exp_name):
        os.mkdir(args.exp_name)
        with open(os.path.join(args.exp_name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

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

    for mamaml_id in range(10):
        save_dir = f"{args.exp_name}/id{mamaml_id}"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

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

        process_args = (args.game, "MAMAML", args.entropy, save_dir, args.checkpoint, mamaml_id, device)
        print(f"RUNNING NAME: {save_dir}")
        processes[device_idx] = mp.Process(target=main, args=process_args)
        processes[device_idx].start()
