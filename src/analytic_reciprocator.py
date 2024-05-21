from collections import deque

import torch

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(precision=4, sci_mode=False)


def idx_to_state(idx: int or torch.Tensor):
    return torch.tensor([idx % 2, idx // 2], dtype=torch.long)


class AnalyticReciprocator:
    def __init__(self, rr_weight: float, buffer_size: int, target_period: int, gamma: float, bsz: int,
                 game: str, device: torch.device):
        """0 is cooperate, 1 is defect"""
        self.own_baseline_policy_buffer = deque(maxlen=buffer_size)
        self.opponent_baseline_policy_buffer = deque(maxlen=buffer_size)
        self.own_baseline_policy = torch.sigmoid(torch.zeros((bsz, 5)).to(device))  # (bsz, 5)
        self.opponent_baseline_policy = torch.sigmoid(torch.zeros((bsz, 5)).to(device))  # (bsz, 5)
        self.own_baseline_policy_buffer.append(self.own_baseline_policy)
        self.opponent_baseline_policy_buffer.append(self.opponent_baseline_policy)

        self.rr_weight = rr_weight
        self.buffer_size = buffer_size
        self.target_period = target_period
        self.gamma = gamma
        self.bsz = bsz
        self.device = device
        # Payoff depends on the s_t-1, s_t, and a_t
        #  After 2 steps, 4 possible states for t-1, 4 possible states for s_t, 4 possible combos of a_t (i.e. s_t+1)
        if game == "IPD":
            self.extrinsic_rewards = torch.Tensor([[-1, -3], [0, -2]]).to(device)
        elif game == "chicken":
            self.extrinsic_rewards = torch.Tensor([[0, -1], [1, -100]]).to(device)
        else:
            raise ValueError("Game not recognized.")

        self.grudge = torch.zeros(bsz, 4, 4).to(device)  # (bsz, 4, 4), dim 1 is s_pre and dim 2 is s
        self.voi_on_other = torch.zeros(bsz, 4, 4).to(device)  # (bsz, 4, 4), dim 1 is s and dim 2 is a
        self.full_rewards = torch.zeros(self.bsz, 4, 4, 4).to(device)

        # RR will be the product of these two
        self.init_full_rewards()
        self.episode_count = 0

    def reset(self):
        self.own_baseline_policy_buffer = deque(maxlen=self.buffer_size)
        self.opponent_baseline_policy_buffer = deque(maxlen=self.buffer_size)
        self.own_baseline_policy = torch.sigmoid(torch.zeros((self.bsz, 5)).to(self.device))  # (bsz, 5)
        self.opponent_baseline_policy = torch.sigmoid(torch.zeros((self.bsz, 5)).to(self.device))  # (bsz, 5)
        self.own_baseline_policy_buffer.append(self.own_baseline_policy)
        self.opponent_baseline_policy_buffer.append(self.opponent_baseline_policy)
        self.grudge = torch.zeros(self.bsz, 4, 4).to(self.device)  # (bsz, 4, 4), dim 1 is s_pre and dim 2 is s
        self.voi_on_other = torch.zeros(self.bsz, 4, 4).to(self.device)  # (bsz, 4, 4), dim 1 is s and dim 2 is a
        self.full_rewards = torch.zeros(self.bsz, 4, 4, 4).to(self.device)

        # RR will be the product of these two
        self.init_full_rewards()
        self.episode_count = 1

    def update_baseline(self, th, tau: float = 1.0):
        self.own_baseline_policy_buffer.append(torch.sigmoid(th[0]))
        self.opponent_baseline_policy_buffer.append(
            torch.sigmoid(th[1][:, torch.LongTensor([0, 1, 3, 2, 4]).to(self.device)]))

        if self.episode_count % self.target_period == 0:
            target_own_baseline_policy = torch.mean(torch.stack(list(self.own_baseline_policy_buffer)), dim=0)
            target_opponent_baseline_policy = torch.mean(torch.stack(list(self.opponent_baseline_policy_buffer)), dim=0)
            self.own_baseline_policy = target_own_baseline_policy * tau + self.own_baseline_policy * (1. - tau)
            self.opponent_baseline_policy = target_opponent_baseline_policy * tau + self.opponent_baseline_policy * (
                        1. - tau)
            self.init_full_rewards()
        # print(torch.sigmoid(th[0][0]))
        # print(torch.sigmoid(th[1][0]))
        # print("RR policy")
        # print(torch.sigmoid(torch.max(th[0], dim=0)[0]), torch.sigmoid(torch.max(th[0], dim=0)[0]))
        # print(torch.sigmoid(torch.max(th[0], dim=0)[0]), torch.sigmoid(torch.max(th[0], dim=0)[0]))
        # print("MFOS policy")
        # print(torch.sigmoid(torch.max(th[1], dim=0)[0]), torch.sigmoid(torch.max(th[1], dim=0)[0]))
        # print(torch.sigmoid(torch.min(th[1], dim=0)[0]), torch.sigmoid(torch.min(th[1], dim=0)[0]))
        # print("RR COMPONENTS")
        # print(self.grudge[0])
        # print(self.voi_on_other[0])

    def Ls(self, th):
        """
        Compute the loss function for the agent.
        :param th: List of 2 tensors of shape (bsz, 5) representing the policy parameters.
        """
        self.episode_count += 1
        # Prob of cooperating in starting state s_0
        p0_init = torch.sigmoid(th[0][:, 0:1])
        p1_init = torch.sigmoid(th[1][:, 0:1])

        # Remove initial state from policy
        p0 = torch.sigmoid(th[0][:, 1:]).view(self.bsz, 4, 1)
        # Permute agent 1's egocentric policy (i.e. its state is flipped) to match agent 0's perspective
        p1 = torch.sigmoid(th[1][:, torch.LongTensor([1, 3, 2, 4]).to(self.device)]).view(self.bsz, 4, 1)

        # TODO: Compute initial state vector
        # Initial opening state combos (bsz, 4)
        S1 = torch.cat([p0_init * p1_init,  # P(CC | s_0)
                        p0_init * (1 - p1_init),  # P(CD | s_0)
                        (1 - p0_init) * p1_init,  # P(DC | s_0)
                        (1 - p0_init) * (1 - p1_init)], dim=-1)  # P(DD | s_0)
        S3 = torch.zeros(self.bsz, 4, 4, 4).to(self.device)

        # Probability of transitioning to each state CD from the current state AB - this is independent since memory-1
        #  equal to p(a_t | s_t), since a_t = s_t+1
        T1 = torch.cat([p0 * p1, p0 * (1 - p1), (1 - p0) * p1, (1 - p0) * (1 - p1)], dim=-1)  # (bsz, 4, 4)
        # P(s_2 | s_1), e.g. T1[0, 1] = P(CC | CD)
        S2 = torch.zeros(self.bsz, 4, 4).to(self.device)

        for s_pre in range(4):
            for s in range(4):
                S2[:, s_pre, s] = S1[:, s_pre] * T1[:, s_pre, s]

        # Probability of transitioning from compound state ABCDEF to GHIJKL (s_pre, s, a) to (s, a, a_next)
        T2 = torch.zeros(self.bsz, 4, 4, 4, 4, 4, 4).to(self.device)
        for s_pre in range(4):
            for s in range(4):
                for a in range(4):
                    S3[:, s_pre, s, a] = S2[:, s_pre, s] * T1[:, s, a]
                    for a_next in range(4):
                        T2[:, s_pre, s, a, s, a, a_next] = T1[:, a, a_next]  # Remember is only memory-1

        S3 = S3.view(self.bsz, -1)  # (bsz, 64)
        T2 = T2.view(self.bsz, 64, 64)
        M = torch.matmul(S3.unsqueeze(1), torch.inverse(torch.eye(64).to(self.device) - self.gamma * T2))
        L_rr = -torch.matmul(M, torch.reshape(self.full_rewards, (self.bsz, 64, 1)).detach())
        return L_rr.squeeze(-1)

    def state_to_choices(self, s: int) -> str:
        choices = ['C', 'D']
        s_idxs = idx_to_state(s)
        return choices[s_idxs[0]] + choices[s_idxs[1]]

    def init_rr_components(self, s_pre: int, s: int, a: int, verbose: bool = False):
        """
        Compute the components (grudge and VoI on other) needed to compute reciprocal rewards.
        :param s_pre: The state at t-1 as an int (0-3)
        :param s: The state at t as an int
        :param a: The action at t as an int
        """
        if verbose:
            print("STATE:", self.state_to_choices(s_pre), self.state_to_choices(s), self.state_to_choices(a))
        # Compute grudge
        s_state = idx_to_state(s)
        last_rew = self.extrinsic_rewards[s_state[0], s_state[1]]  # Actual reward received at t-1 (since s_t is a_t-1)
        baseline_probs = self.opponent_baseline_policy[:, s_pre]  # (bsz,) p(cooperate | s_pre/t-1)
        # P(C) * r(rc's actual action, C) + P(D) * r(rc's actual action, D) at t-1
        last_expected_rew = (self.extrinsic_rewards[s_state[0], 0] * baseline_probs +
                             self.extrinsic_rewards[s_state[0], 1] * (1 - baseline_probs))
        self.grudge[:, s_pre, s] = last_expected_rew - last_rew  # essentially just VoI on self for 1-step memory
        if verbose:
            # print(f"OPP BASELINE P(C | {self.state_to_choices(s_pre)}):", baseline_probs[0])
            # print(f"ACTUAL REWARD R({self.state_to_choices(s)} | {self.state_to_choices(s_pre)}):", last_rew,
            #       f"EXPECTED REW FOR SELF R({self.state_to_choices(s)[0]}X | {self.state_to_choices(s_pre)}):", last_expected_rew[0])
            print("GRUDGE:", self.grudge[0, s_pre, s])

        # Compute VoI on other
        a_state = idx_to_state(a)
        curr_opp_rew = self.extrinsic_rewards[
            a_state[1], a_state[0]]  # Extrinsic rew at current time t from actions a_t
        own_baseline_probs = self.own_baseline_policy[:, s]  # (bsz,)
        opp_expected_rew = (self.extrinsic_rewards[a_state[1], 0] * own_baseline_probs +
                            self.extrinsic_rewards[a_state[1], 1] * (1 - own_baseline_probs))
        self.voi_on_other[:, s, a] = opp_expected_rew - curr_opp_rew
        if verbose:
            # print(f"OWN BASELINE PROBS P(C | {self.state_to_choices(s)}):", own_baseline_probs[0])
            # print(f"ACTUAL REWARD R({self.state_to_choices(a)} | {self.state_to_choices(s)}):", curr_opp_rew,
            #       f"EXPECTED REWARD FOR OPP R(X{self.state_to_choices(a)[1]} | {self.state_to_choices(s)}):", opp_expected_rew[0])
            print("VOI:", self.voi_on_other[0, s, a])
            print("RR:", self.voi_on_other[0, s, a] * self.grudge[0, s_pre, s])

    def init_full_rewards(self):
        for s_pre in range(4):
            for s in range(4):
                for a in range(4):
                    self.init_rr_components(s_pre, s, a)
        # print("GRUDGE")
        # print(self.grudge.max(), self.grudge.min())
        # print(self.voi_on_other.max(), self.voi_on_other.min())
        for s_pre in range(4):
            for s in range(4):
                for a in range(4):
                    a_state = idx_to_state(a)
                    self.full_rewards[:, s_pre, s, a] = self.grudge[:, s_pre, s] * self.voi_on_other[:, s,
                                                                                   a] * self.rr_weight
                    self.full_rewards[:, s_pre, s, a] += self.extrinsic_rewards[a_state[0], a_state[1]]
