import argparse
import json
import os

import torch

from src.main_mfos_ppo import main


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

    for mamaml_id in range(10):
        device = torch.device(args.device)
        exp_name = f"{args.exp_name}/id_{mamaml_id}"
        if not os.path.isdir(exp_name):
            os.mkdir(exp_name)
        print(f"RUNNING NAME: {exp_name}")
        main(args.game, "MAMAML", args.entropy, exp_name, args.checkpoint, mamaml_id, device)
