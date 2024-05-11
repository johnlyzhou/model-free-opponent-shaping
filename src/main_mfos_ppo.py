import torch
from ppo import PPO, Memory
from environments import MetaGames
import os
import argparse
import json
from tqdm import tqdm
torch.set_printoptions(precision=4, sci_mode=False)

parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, required=True)
parser.add_argument("--opponent", type=str, required=True)
parser.add_argument("--entropy", type=float, default=0.01)
parser.add_argument("--exp-name", type=str, default="test")
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--mamaml-id", type=int, default=0)
args = parser.parse_args()


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    ############################################
    K_epochs = 4  # update policy for K epochs

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.002  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    max_episodes = 1024
    if args.checkpoint:
        max_episodes = 1
        batch_size = 2
    else:
        batch_size = 4096
    random_seed = None
    num_steps = 100

    save_freq = 100
    name = args.exp_name

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(name):
        os.mkdir(name)
        with open(os.path.join(name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    #############################################

    # creating environment
    env = MetaGames(batch_size, opponent=args.opponent, game=args.game, mmapg_id=args.mamaml_id)

    action_dim = env.d  # 5 for IPD
    state_dim = env.d * 2  # concatenation of both policies so 10
    print(f"action_dim: {action_dim}, state_dim: {state_dim}")

    # This is actually the outer MFOS agent that is outputting a 5-d policy to play against the inner PPO agent
    #  at each step, i.e., a full inner episode
    memory = Memory()
    ppo = PPO(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, args.entropy)

    if args.checkpoint:
        ppo.load(args.checkpoint)

    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    rew_means = []
    logs = []

    for i_episode in tqdm(range(1, max_episodes + 1)):
        state = env.reset()

        running_reward = torch.zeros(batch_size).to(device)
        running_opp_reward = torch.zeros(batch_size).to(device)

        last_reward = 0
        last_opp_reward = 0

        for t in tqdm(range(num_steps)):
            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            # Env step gives the full rewards of an entire inner episode between the two agents
            state, reward, info, M = env.step(action)
            if args.checkpoint or t == 0 or t == num_steps - 1:
                print(f"EP {i_episode}, Step {t}")
                print(f"Mean MFOS policy: {state[:, :5].mean(dim=0)}\n"
                      f"Std MFOS policy: {state[:, :5].std(dim=0)}\n")
                print(f"Mean RR policy: {state[:, 5:].mean(dim=0)}\n"
                      f"Std RR policy: {state[:, 5:].std(dim=0)}\n")

            memory.rewards.append(reward)
            running_reward += reward.squeeze(-1)
            running_opp_reward += info.squeeze(-1)
            last_reward = reward.squeeze(-1).mean().item()
            last_opp_reward = info.squeeze(-1).mean().item()
            # print(last_reward.shape, last_opp_reward.shape)

        if not args.checkpoint:
            ppo.update(memory)
        memory.clear_memory()

        print("=" * 100, flush=True)

        print(f"episode: {i_episode}", flush=True)

        print(f"loss: {-running_reward.mean() / num_steps}", flush=True)

        rew_means.append(
            {
                "rew": (running_reward.mean() / num_steps).item(),
                "opp_rew": (running_opp_reward.mean() / num_steps).item(),
            }
        )

        logs.append({f"player_{0}": {"last_reward": last_reward, "avg_rew": running_reward.mean().item() / num_steps},
                     f"player_{1}": {"last_reward": last_opp_reward, "avg_rew": running_opp_reward.mean().item() / num_steps}})
        print(logs[-1])

        if not args.checkpoint and i_episode % save_freq == 0:
            ppo.save(os.path.join(name, f"{i_episode}.pth"))
            with open(os.path.join(name, f"out_{i_episode}.json"), "w") as f:
                json.dump(logs, f)

    if not args.checkpoint:
        ppo.save(os.path.join(name, f"{i_episode}.pth"))
        with open(os.path.join(name, f"out_{i_episode}.json"), "w") as f:
            json.dump(rew_means, f)
        print(f"SAVING! {i_episode}")
