import torch


def idx_to_state(idx: int or torch.Tensor):
    return torch.tensor([idx % 2, idx // 2], dtype=torch.long)


class AnalyticReciprocator:
    def __init__(self, own_baseline_policy: torch.Tensor, opponent_baseline_policy: torch.Tensor, rr_weight: float,
                 gamma: float, bsz: int, device: torch.device):
        """0 is cooperate, 1 is defect"""
        self.own_baseline_policy = own_baseline_policy  # (bsz, 5)
        self.opponent_baseline_policy = opponent_baseline_policy  # (bsz, 5)
        self.rr_weight = rr_weight
        self.gamma = gamma
        self.bsz = bsz
        self.device = device
        # Payoff depends on the s_t-1, s_t, and a_t
        #  After 2 steps, 4 possible states for t-1, 4 possible states for s_t, 4 possible combos of a_t (i.e. s_t+1)
        self.extrinsic_rewards = torch.Tensor([[-1, -3], [0, -2]]).to(device)
        self.grudge = torch.zeros(bsz, 4, 4).to(device)  # (bsz, 4, 4), dim 1 is s_pre and dim 2 is s
        self.voi_on_other = torch.zeros(bsz, 4, 4).to(device)  # (bsz, 4, 4), dim 1 is s and dim 2 is a
        self.full_rewards = torch.zeros(self.bsz, 4, 4, 4).to(device)

        # RR will be the product of these two
        self.init_full_rewards()

    def update_baseline(self, th, tau: float = 1.0):
        self.own_baseline_policy = th[0] * tau + self.own_baseline_policy * (1. - tau)
        self.opponent_baseline_policy = (th[1] * tau + self.opponent_baseline_policy) * (1. - tau)

    def Ls(self, th):
        """
        Compute the loss function for the agent.
        :param th: List of 2 tensors of shape (bsz, 5) representing the policy parameters.
        """
        # Prob of cooperating in starting state s_0
        p0_init = torch.sigmoid(th[0][:, 0:1])
        p1_init = torch.sigmoid(th[1][:, 0:1])

        # Remove initial state from policy
        p0 = torch.sigmoid(th[0][:, 1:]).view(self.bsz, 4, 1)
        # Permute agent 1's egocentric policy (i.e. its state is flipped) to match agent 0's perspective
        p1 = torch.sigmoid(th[1][:, torch.LongTensor([1, 3, 2, 4]).to(self.device)]).view(self.bsz, 4, 1)

        # TODO: Compute initial state vector
        # Initial opening state combos
        S1 = torch.cat([p0_init * p1_init,
                        p0_init * (1 - p1_init),
                        (1 - p0_init) * p1_init,
                        (1 - p0_init) * (1 - p1_init)], dim=-1)  # (bsz, 4)
        S3 = torch.zeros(self.bsz, 4, 4, 4).to(self.device)

        # Probability of transitioning to each state CD from the current state AB - this is independent since memory-1
        #  equal to p(a_t | s_t), since a_t = s_t+1
        T1 = torch.cat([p0 * p1, p0 * (1 - p1), (1 - p0) * p1, (1 - p0) * (1 - p1)], dim=-1)  # (bsz, 4, 4)
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
        L_rr = -torch.matmul(M, torch.reshape(self.full_rewards, (self.bsz, 64, 1)))
        return L_rr.squeeze(-1)

    def init_rr_components(self, s_pre: int, s: int, a: int):
        """
        Compute the components (grudge and VoI on other) needed to compute reciprocal rewards.
        :param s_pre: The state at t-1 as an int (0-3)
        :param s: The state at t as an int
        :param a: The action at t as an int
        """
        s_state = idx_to_state(s)
        last_rew = self.extrinsic_rewards[s_state[0], s_state[1]]  # Actual reward received at t-1 (since s_t is a_t-1)
        baseline_probs = self.opponent_baseline_policy[:, s_pre]  # (bsz,) p(cooperate | s_pre/t-1)
        # P(C) * r(rc's actual action, C) + P(D) * r(rc's actual action, D) at t-1
        last_expected_rew = (self.extrinsic_rewards[s_state[0], 0] * baseline_probs +
                             self.extrinsic_rewards[s_state[0], 1] * (1 - baseline_probs))
        self.grudge[:, s_pre, s] = last_expected_rew - last_rew  # essentially just VoI on self for 1-step memory

        a_state = idx_to_state(a)
        curr_rew = self.extrinsic_rewards[a_state[0], a_state[1]]  # Extrinsic rew at current time t from actions a_t
        baseline_probs = self.own_baseline_policy[:, s]  # (bsz,)
        opp_expected_rew = (self.extrinsic_rewards[a_state[1], 0] * baseline_probs +
                            self.extrinsic_rewards[a_state[1], 1] * (1 - baseline_probs))
        self.voi_on_other[:, s, a] = opp_expected_rew - curr_rew

    def init_full_rewards(self):
        self.full_rewards = torch.zeros(self.bsz, 4, 4, 4).to(self.device)
        for s_pre in range(4):
            for s in range(4):
                for a in range(4):
                    self.init_rr_components(s_pre, s, a)
                    a_state = idx_to_state(a)
                    self.full_rewards[:, s_pre, s, a] = self.grudge[:, s_pre, s] * self.voi_on_other[:, s, a] * self.rr_weight
                    self.full_rewards[:, s_pre, s, a] += self.extrinsic_rewards[a_state[0], a_state[1]]
