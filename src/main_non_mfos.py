import os
import json
import torch
import argparse
from environments import NonMfosMetaGames

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default="")
args = parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 8192
    num_steps = 100
    name = args.exp_name

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(name):
        os.mkdir(name)
        with open(os.path.join(name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    results = []
    for game in ["IPD", "chicken"]:
        for p1 in ["Reciprocator"]:  #"Reciprocator", "NL", "LOLA"]:
            for p2 in ["NL", "LOLA", "MAMAML"]:
                if p1 == "MAMAML" or p2 == "MAMAML":
                    for id in range(10):
                        print(f"Running {game} with {p1} vs. {p2}: ID {id}")
                        env = NonMfosMetaGames(batch_size, game=game, p1=p1, p2=p2, mmapg_id=id)
                        env.reset()
                        running_rew_0 = torch.zeros(batch_size).to(device)
                        running_rew_1 = torch.zeros(batch_size).to(device)
                        for i in tqdm(range(num_steps)):
                            _, r0, r1, M = env.step()
                            running_rew_0 += r0.squeeze(-1)
                            running_rew_1 += r1.squeeze(-1)
                        mean_rew_0 = (running_rew_0.mean() / num_steps).item()
                        mean_rew_1 = (running_rew_1.mean() / num_steps).item()

                        results.append(
                            {
                                "game": game,
                                "p1": p1,
                                "p2": p2,
                                "rew_0": mean_rew_0,
                                "rew_1": mean_rew_1,
                                "mmapg_id": id,
                            }
                        )
                        print("=" * 100)
                        print(f"Done with game: {game}, p1: {p1}, p2: {p2}, id: {id}")
                        print(f"r0: {mean_rew_0}, r1: {mean_rew_1}")
                else:
                    print(f"Running {game} with {p1} vs. {p2}")
                    env = NonMfosMetaGames(batch_size, game=game, p1=p1, p2=p2)
                    env.reset()
                    running_rew_0 = torch.zeros(batch_size).to(device)
                    running_rew_1 = torch.zeros(batch_size).to(device)
                    for i in tqdm(range(num_steps)):
                        state, r0, r1, M = env.step()
                        print(f"Step {i}")
                        print(f"Mean {p1} policy: {state[:, :5].mean(dim=0)}\n"
                              f"Std {p1} policy: {state[:, :5].std(dim=0)}\n")
                        print(f"Mean {p2} policy: {state[:, 5:].mean(dim=0)}\n"
                              f"Std {p2} policy: {state[:, 5:].std(dim=0)}\n")
                        running_rew_0 += r0.squeeze(-1)
                        running_rew_1 += r1.squeeze(-1)
                    mean_rew_0 = (running_rew_0.mean() / num_steps).item()
                    mean_rew_1 = (running_rew_1.mean() / num_steps).item()

                    results.append({"game": game, "p1": p1, "p2": p2, "rew_0": mean_rew_0, "rew_1": mean_rew_1})
                    print("=" * 100)
                    print(f"Done with game: {game}, p1: {p1}, p2: {p2}")
                    print(f"r0: {mean_rew_0}, r1: {mean_rew_1}")
    with open(os.path.join(name, f"out.json"), "w") as f:
        json.dump(results, f)
