"""
SAC (Soft Actor-Critic) — Pendulum-v1 完整训练 + 学习曲线
连续控制 + 最大熵 + Double Q + 自动温度 + 评测 + 画图

运行: pip install torch gym matplotlib
      python code/sac_pendulum.py      (约 2 分钟出图)
"""
import gym, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt
from collections import deque
import random

# === 1. Actor（高斯策略）===
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit=2.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU())
        self.mu = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))
        self.act_limit = act_limit

    def forward(self, obs):
        h = self.net(obs)
        std = self.log_std.exp().clamp(1e-4, 10)
        dist = torch.distributions.Normal(self.mu(h), std)
        pi = dist.rsample()
        log_prob = dist.log_prob(pi).sum(-1)
        return torch.tanh(pi) * self.act_limit, log_prob

# === 2. Critic（Double Q）===
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(obs_dim + act_dim, 128), nn.ReLU(),
                                nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.q2 = nn.Sequential(nn.Linear(obs_dim + act_dim, 128), nn.ReLU(),
                                nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.q1(x), self.q2(x)

# === 3. 经验池 ===
class Buffer:
    def __init__(self, size=100000):
        self.buf = deque(maxlen=size)
    def add(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))
    def sample(self, n=256):
        batch = random.sample(self.buf, min(n, len(self.buf)))
        return [torch.tensor(np.array(x), dtype=torch.float32) for x in zip(*batch)]
    def __len__(self):
        return len(self.buf)

# === 4. SAC ===
class SAC:
    def __init__(self, obs_dim, act_dim):
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim, act_dim)
        self.target = Critic(obs_dim, act_dim)
        self.target.load_state_dict(self.critic.state_dict())

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -act_dim

        self.buf = Buffer()
        self.gamma = 0.99
        self.tau = 0.005

    def act(self, obs, eval=False):
        with torch.no_grad():
            pi, _ = self.actor(torch.tensor(obs, dtype=torch.float32))
            return pi.numpy()

    def update(self):
        if len(self.buf) < 256: return
        s, a, r, ns, d = self.buf.sample(256)

        # Critic
        with torch.no_grad():
            na, nlp = self.actor(ns)
            nq1, nq2 = self.target(ns, na)
            nq = torch.min(nq1, nq2) - self.log_alpha.exp() * nlp.unsqueeze(-1)
            target_q = r.unsqueeze(-1) + self.gamma * (1 - d.unsqueeze(-1)) * nq
        q1, q2 = self.critic(s, a)
        loss_c = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()

        # Actor
        pi, lp = self.actor(s)
        q1_pi, q2_pi = self.critic(s, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_a = (self.log_alpha.exp() * lp - q_pi.squeeze(-1)).mean()
        self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()

        # Alpha
        _, lp2 = self.actor(s)
        loss_alpha = -(self.log_alpha * (lp2 + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad(); loss_alpha.backward(); self.opt_alpha.step()

        # Soft update target
        for tp, p in zip(self.target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

# === 5. 评测 ===
def evaluate(agent, env, episodes=5):
    total = 0
    for _ in range(episodes):
        s, done = env.reset()[0], False
        ep_r = 0
        while not done:
            s, r, done, _, _ = env.step(agent.act(s, eval=True))
            ep_r += r
        total += ep_r
    return total / episodes

# === 6. 训练 ===
def train(steps=50000):
    env = gym.make("Pendulum-v1")
    agent = SAC(env.observation_space.shape[0], env.action_space.shape[0])
    eval_rewards = []

    s = env.reset()[0]
    for step in range(steps):
        a = agent.act(s)
        ns, r, done, _, _ = env.step(a)
        agent.buf.add(s, a, r, ns, float(done))
        agent.update()
        s = ns if not done else env.reset()[0]

        if step % 5000 == 0 and step > 0:
            avg = evaluate(agent, env)
            eval_rewards.append((step, avg))
            print(f"[SAC] Step {step}: eval_reward={avg:.1f}")

    env.close()
    return eval_rewards

if __name__ == "__main__":
    print("SAC on Pendulum-v1")
    rewards = train()
    steps, vals = zip(*rewards)
    plt.plot(steps, vals, "-o")
    plt.xlabel("Training Step"); plt.ylabel("Avg Reward (5 episodes)")
    plt.title("SAC on Pendulum-v1"); plt.grid(); plt.savefig("sac_result.png")
    print("Saved sac_result.png")
