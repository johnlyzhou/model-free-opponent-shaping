import pathlib

import torch
import os
import json
from src.coin_game.coin_game_envs import CoinGamePPO
from src.coin_game.coin_game_mfos_agent import MemoryMFOS, PPOMFOS
import argparse


def main_mfos_coin_game(save_dir, device):
    batch_size = 512  # 8192 #, 32768
    state_dim = [7, 3, 3]
    action_dim = 4
    n_latent_var = 16  # number of variables in hidden layer
    max_episodes = 1000  # max training episodes
    log_interval = 50

    lr = 0.0002
    betas = (0.9, 0.999)
    gamma = 0.995  # discount factor
    tau = 0.3  # GAE

    traj_length = 16

    K_epochs = 16  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    use_gae = False

    inner_ep_len = 32
    num_steps = inner_ep_len * traj_length

    do_sum = False

    save_freq = 50

    #############################################

    memory = MemoryMFOS()
    ppo = PPOMFOS(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, batch_size, inner_ep_len,
                  device)
    print(lr, betas)
    print(sum(p.numel() for p in ppo.policy_old.parameters() if p.requires_grad))
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    rew_means = []
    outer_logs = []

    # env
    env = CoinGamePPO(batch_size, inner_ep_len, device, save_dir=save_dir)

    # training loop
    for i_episode in range(1, max_episodes + 1):
        memory.clear_memory()
        state = env.reset()  # Reset environment, including NL PPO opponent back to initialization
        # These stats are kept across the meta-episode which consists of multiple episodes of CoinGame -
        #  should normalize this by the number of episodes in the meta-episode or just record outcomes of last one
        running_reward = 0
        opp_running_reward = 0
        p1_num_opp, p2_num_opp, p1_num_self, p2_num_self = 0, 0, 0, 0
        # Num_steps is the total number of steps num_episodes * inner_episode_len
        # Each t is an actual step of CoinGame
        for t in range(num_steps):
            # Running policy_old:
            if t % inner_ep_len == 0:
                # If t == 0, then this is the start of a meta-episode (new i_episode) and you fully reset the
                #  conditioning vector th_ba
                #  else if just t % inner_ep_len == 0, then this is the start of a new inner episode and you generate a
                #  new th_ba conditoning vector based on your meta-policy
                ppo.policy_old.reset(memory, t == 0)
            with torch.no_grad():
                action = ppo.policy_old.act(state.detach())  # Get an action from MFOS actor
            state, reward, done, info, info_2 = env.step(action.detach())
            running_reward += reward.detach()
            opp_running_reward += info.detach()
            memory.rewards.append(reward.detach())
            if info_2 is not None:
                p1_num_opp += info_2[2]
                p2_num_opp += info_2[1]
                p1_num_self += info_2[3]
                p2_num_self += info_2[0]

        env.logs.log_meta_episode(save=False)
        outer_logs.append(env.logs.outer_logs[-1])

        ppo.policy_old.reset(memory)
        ppo.update(memory)

        print("=" * 10)

        rew_means.append(
            {
                "episode": i_episode,
                "rew": running_reward.mean().item(),
                "opp_rew": opp_running_reward.mean().item(),
                "p1_opp": p1_num_opp.float().mean().item(),
                "p2_opp": p2_num_opp.float().mean().item(),
                "p1_self": p1_num_self.float().mean().item(),
                "p2_self": p2_num_self.float().mean().item(),
            }
        )
        print(rew_means[-1])

        old_log_path = os.path.join(save_dir, "old")
        if not os.path.isdir(old_log_path):
            pathlib.Path(old_log_path).mkdir(parents=True, exist_ok=True)

        if i_episode % save_freq == 0 or i_episode == max_episodes:
            ppo.save(os.path.join(old_log_path, f"{i_episode}.pth"))
            with open(os.path.join(save_dir, f"out_{i_episode}.json"), "w") as f:
                json.dump(outer_logs, f)
            with open(os.path.join(old_log_path, f"out_{i_episode}.json"), "w") as f:
                json.dump(rew_means, f)
            print(f"SAVING! {i_episode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    name = args.exp_name

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(name):
        os.mkdir(name)
        with open(os.path.join(name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    main_mfos_coin_game(name, device)
