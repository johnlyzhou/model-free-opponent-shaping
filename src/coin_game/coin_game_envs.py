import json
import os

import torch
import numpy as np
from src.coin_game.coin_game_ppo_agent import PPO, Memory


class CoinGameStats:
    def __init__(self, device: torch.device, num_players: int = 2, save_dir: str = 'my_logs', save_interval: int = 50):
        self.episode_count = 0
        self.meta_episode_count = 1
        self.num_players = num_players
        self.device = device
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.batch_size = None
        self.running_rewards_per_batch = torch.zeros(num_players, device=device)
        self.own_coin_count = torch.zeros(num_players, device=device)
        self.other_coin_count = torch.zeros(num_players, device=device)
        self.total_coin_count = torch.zeros(num_players, device=device)
        self.logs = []
        self.outer_logs = []

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def update(self, rewards: list, info: tuple):
        """
        Update the running statistics of the environment.
        :param rewards: List of length num_players of tensors of shape (batch_size,).
        :param info: Tuple containing the total coin collection counts across the batch (red_coins_collected_by_red,
        blue_coins_collected_by_red, red_coins_collected_by_blue, blue_coins_collected_by_blue) - 0: red, 1: blue.
        """
        if self.batch_size is None:
            self.batch_size = rewards[0].shape[0]

        self.own_coin_count[0] += info[0].item() / self.batch_size
        self.other_coin_count[0] += info[1].item() / self.batch_size
        self.own_coin_count[1] += info[3].item() / self.batch_size
        self.other_coin_count[1] += info[2].item() / self.batch_size
        self.total_coin_count[0] += (info[0] + info[1]).item() / self.batch_size
        self.total_coin_count[1] += (info[2] + info[3]).item() / self.batch_size

        for agent_idx in range(self.num_players):
            self.running_rewards_per_batch[agent_idx] += rewards[agent_idx].mean().item()

    def _log_episode_agent(self, idx):
        agent_logs = {f"reward": self.running_rewards_per_batch[idx].item(),
                      "own_coin_count": self.own_coin_count[idx].item(),
                      "other_coin_count": self.other_coin_count[idx].item(),
                      "total_coin_count": self.total_coin_count[idx].item()}
        return agent_logs

    def log_episode(self, verbose: bool = False):
        log = {f"player_{idx}": self._log_episode_agent(idx) for idx in range(self.num_players)}
        self.logs.append(log)
        print(f"EPISODE {self.episode_count}:", self.logs[-1]) if verbose else None
        self._reset()
        self.episode_count += 1

    def log_meta_episode(self, save: bool = True):
        if len(self.logs) == 0:
            print("No episodes to log.")
            return
        self.outer_logs.append(self.logs)
        self.logs = []
        self._reset()
        self.episode_count = 0
        print(f"META-EPISODE {self.meta_episode_count} complete!")
        if self.meta_episode_count % self.save_interval == 0:
            self.save()
        self.meta_episode_count += 1

    def _reset(self):
        self.running_rewards_per_batch = torch.zeros(self.num_players, device=self.device)
        self.own_coin_count = torch.zeros(self.num_players, device=self.device)
        self.other_coin_count = torch.zeros(self.num_players, device=self.device)
        self.total_coin_count = torch.zeros(self.num_players, device=self.device)

    def reset(self):
        self._reset()
        self.logs = []
        self.outer_logs = []

    def save(self):
        with open(os.path.join(self.save_dir, f"out_{self.meta_episode_count}.json"), "w") as f:
            json.dump(self.outer_logs, f)


class CoinGameGPU:
    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """

    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    moves = torch.stack(
        [
            torch.LongTensor([0, 1]),
            torch.LongTensor([0, -1]),
            torch.LongTensor([1, 0]),
            torch.LongTensor([-1, 0]),
        ],
        dim=0,
    )

    def __init__(self, max_steps, batch_size, device, grid_size=3):
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.device = device
        self.MOVES = self.moves.to(device)
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [4, grid_size, grid_size]
        self.NUM_STATES = np.prod(self.ob_space_shape)
        self.available_actions = 4
        self.step_count = None

    def reset(self):
        self.step_count = 0

        red_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(self.device)
        self.red_pos = torch.stack((red_pos_flat // self.grid_size, red_pos_flat % self.grid_size), dim=-1)

        blue_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(self.device)
        self.blue_pos = torch.stack((blue_pos_flat // self.grid_size, blue_pos_flat % self.grid_size), dim=-1)

        red_coin_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(self.device)
        blue_coin_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(self.device)

        self.red_coin_pos = torch.stack((red_coin_pos_flat // self.grid_size, red_coin_pos_flat % self.grid_size),
                                        dim=-1)
        self.blue_coin_pos = torch.stack((blue_coin_pos_flat // self.grid_size, blue_coin_pos_flat % self.grid_size),
                                         dim=-1)

        state = self._generate_state()
        observations = [state, state]
        return observations

    def _generate_coins(self):
        mask_red = torch.logical_or(self._same_pos(self.red_coin_pos, self.blue_pos),
                                    self._same_pos(self.red_coin_pos, self.red_pos))
        red_coin_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(self.device)[mask_red]
        self.red_coin_pos[mask_red] = torch.stack(
            (red_coin_pos_flat // self.grid_size, red_coin_pos_flat % self.grid_size), dim=-1)

        mask_blue = torch.logical_or(self._same_pos(self.blue_coin_pos, self.blue_pos),
                                     self._same_pos(self.blue_coin_pos, self.red_pos))
        blue_coin_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(self.device)[
            mask_blue]
        self.blue_coin_pos[mask_blue] = torch.stack(
            (blue_coin_pos_flat // self.grid_size, blue_coin_pos_flat % self.grid_size), dim=-1)

    def _same_pos(self, x, y):
        return torch.all(x == y, dim=-1)

    def _generate_state(self):
        red_pos_flat = self.red_pos[:, 0] * self.grid_size + self.red_pos[:, 1]
        blue_pos_flat = self.blue_pos[:, 0] * self.grid_size + self.blue_pos[:, 1]

        red_coin_pos_flat = self.red_coin_pos[:, 0] * self.grid_size + self.red_coin_pos[:, 1]
        blue_coin_pos_flat = self.blue_coin_pos[:, 0] * self.grid_size + self.blue_coin_pos[:, 1]

        state = torch.zeros((self.batch_size, 4, self.grid_size * self.grid_size)).to(self.device)

        state[:, 0].scatter_(1, red_pos_flat[:, None], 1)
        state[:, 1].scatter_(1, blue_pos_flat[:, None], 1)
        state[:, 2].scatter_(1, red_coin_pos_flat[:, None], 1)
        state[:, 3].scatter_(1, blue_coin_pos_flat[:, None], 1)

        return state.view(self.batch_size, 4, self.grid_size, self.grid_size)

    def step(self, actions):
        ac0, ac1 = actions

        self.step_count += 1

        self.red_pos = (self.red_pos + self.MOVES[ac0]) % self.grid_size
        self.blue_pos = (self.blue_pos + self.MOVES[ac1]) % self.grid_size

        # Compute rewards
        red_reward = torch.zeros(self.batch_size).to(self.device)
        red_red_matches = self._same_pos(self.red_pos, self.red_coin_pos)
        red_reward[red_red_matches] += 1
        red_blue_matches = self._same_pos(self.red_pos, self.blue_coin_pos)
        red_reward[red_blue_matches] += 1

        blue_reward = torch.zeros(self.batch_size).to(self.device)
        blue_red_matches = self._same_pos(self.blue_pos, self.red_coin_pos)
        blue_reward[blue_red_matches] += 1
        blue_blue_matches = self._same_pos(self.blue_pos, self.blue_coin_pos)
        blue_reward[blue_blue_matches] += 1

        red_reward[blue_red_matches] -= 2
        blue_reward[red_blue_matches] -= 2

        self._generate_coins()
        reward = [red_reward.float(), blue_reward.float()]
        state = self._generate_state()
        observations = [state, state]
        if self.step_count >= self.max_steps:
            done = torch.ones(self.batch_size).to(self.device)
        else:
            done = torch.zeros(self.batch_size).to(self.device)

        return observations, reward, done, (
        red_red_matches.sum(), red_blue_matches.sum(), blue_red_matches.sum(), blue_blue_matches.sum())


class SymmetricCoinGame:
    def __init__(self, b, inner_ep_len, device, gamma_inner=0.96, save_dir: str = "my_logs"):
        self.env = CoinGameGPU(inner_ep_len - 1, b, device)
        self.logs = CoinGameStats(device=torch.device('cpu'), save_dir=save_dir)
        self.inner_ep_len = inner_ep_len
        self.b = b
        self.device = device

    def reset(self):
        self.env_states = self.env.reset()[0]
        self.rewards_inner = torch.Tensor(np.array([0.0] * self.b)).to(self.device)
        self.rewards_outer = torch.Tensor(np.array([0.0] * self.b)).to(self.device)
        self.dones_inner = torch.Tensor(np.array([0.0] * self.b)).to(self.device)
        self.dones_outer = torch.Tensor(np.array([0.0] * self.b)).to(self.device)
        return self._prep_state()

    def _prep_state(self):

        rewards_inner_tiled = torch.tile(self.rewards_inner[None, None].T, [1, 3, 3])[:, None]
        rewards_outer_tiled = torch.tile(self.rewards_outer[None, None].T, [1, 3, 3])[:, None]
        dones_inner_tiled = torch.tile(self.dones_inner[None, None].T, [1, 3, 3])[:, None]
        env_states_outer = torch.stack(
            [self.env_states[:, 1], self.env_states[:, 0], self.env_states[:, 3], self.env_states[:, 2]], dim=1)
        return [
            torch.cat([self.env_states, rewards_inner_tiled, rewards_outer_tiled, dones_inner_tiled], axis=1),
            torch.cat([env_states_outer, rewards_outer_tiled, rewards_inner_tiled, dones_inner_tiled], axis=1),
        ]

    def step(self, actions):
        if torch.any(self.dones_inner):
            info = None
            self.env_states = self.env.reset()[0]
            self.logs.log_episode(verbose=True)
            self.rewards_inner = torch.Tensor(np.array([0.0] * self.b)).to(self.device)
            self.rewards_outer = torch.Tensor(np.array([0.0] * self.b)).to(self.device)
            self.dones_inner = torch.Tensor(np.array([0.0] * self.b)).to(self.device)
        else:
            self.env_states, rewards, self.dones_inner, info = self.env.step(actions)
            self.logs.update(rewards, info)
            self.env_states = self.env_states[0]
            self.rewards_inner, self.rewards_outer = rewards
        return self._prep_state(), [self.rewards_inner, self.rewards_outer], info


class CoinGamePPO:
    def __init__(self, b, inner_ep_len, device, gamma_inner=0.96, first=False, save_dir: str = "my_logs"):
        self.env = CoinGameGPU(inner_ep_len - 1, b, device)
        self.logs = CoinGameStats(device=torch.device('cpu'), save_dir=save_dir)
        self.inner_ep_len = inner_ep_len
        self.b = b
        self.first = first
        self.device = device

    def reset(self):
        """
        Hyperparams for inner PPO
        """
        input_shape = [4, 3, 3]
        action_dim = 4
        n_latent_var = 8
        lr = 0.005

        betas = (0.9, 0.999)
        gamma = 0.96  # discount factor
        tau = 0.3  # GAE
        K_epochs = 80  # update policy for K epochs
        eps_clip = 0.2  # clip parameter for PPO

        self.inner_agent = PPO(input_shape, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip,
                               self.device)  # MAYBE DONT LET INNER AGENT BE RNN
        self.inner_memory = Memory()

        self.env_states = self.env.reset()
        self.logs.log_meta_episode()
        self.rewards_inner = torch.zeros(self.b).to(self.device)
        self.rewards_outer = torch.zeros(self.b).to(self.device)
        self.dones_inner = torch.zeros(self.b).to(self.device)
        self.dones_outer = torch.zeros(self.b).to(self.device)
        self.t = 0

        return self._prep_state()

    def _prep_state(self):
        rewards_inner_tiled = torch.tile(self.rewards_inner[None, None].T, [1, 3, 3])[:, None]
        rewards_outer_tiled = torch.tile(self.rewards_outer[None, None].T, [1, 3, 3])[:, None]
        dones_inner_tiled = torch.tile(self.dones_inner[None, None].T, [1, 3, 3])[:, None]

        return torch.cat([self.env_states[0], rewards_inner_tiled, rewards_outer_tiled, dones_inner_tiled], axis=1)

    def step(self, actions):
        """Actions is just MFOS's action, the inner PPO agent's action is sampled inside."""
        self.t += 1
        if torch.any(self.dones_inner):
            info = None
            assert self.t % self.inner_ep_len == 0
            self.env_states = self.env.reset()
            self.logs.log_episode(verbose=True)
            self.rewards_inner = torch.zeros(self.b).to(self.device)
            self.rewards_outer = torch.zeros(self.b).to(self.device)
            self.dones_inner = torch.zeros(self.b).to(self.device)
            self.dones_outer = torch.zeros(self.b).to(self.device)
            self.inner_agent.update(self.inner_memory)
            self.inner_memory.clear_memory()
        else:
            with torch.no_grad():
                # Sample inner PPO agent's action
                self.inner_actions = self.inner_agent.policy_old.act(self.env_states[1], self.inner_memory)

                if self.first:
                    self.env_states, rewards, self.dones_inner, info = self.env.step([actions, self.inner_actions])
                    self.rewards_outer, self.rewards_inner = rewards
                else:
                    self.env_states, rewards, self.dones_inner, info = self.env.step([self.inner_actions, actions])
                    self.rewards_inner, self.rewards_outer = rewards

                self.inner_memory.rewards.append(self.rewards_inner)
                self.logs.update(rewards, info)

        return self._prep_state(), self.rewards_outer, self.dones_outer, self.rewards_inner, info
