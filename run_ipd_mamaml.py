import argparse

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

    for mamaml_id in range(10):
        device = torch.device(args.device)
        exp_name = f"{args.exp_name}_{mamaml_id}"
        print(f"RUNNING NAME: {exp_name}")
        main(args.game, "MAMAML", args.entropy, exp_name, args.checkpoint, mamaml_id, device)
