"""
DQN vs Double DQN — CartPole-v1 完整训练 + 对比
包含：训练循环、周期性评测、学习曲线、两种算法对比

运行: pip install torch gym matplotlib
      python code/dqn_cartpole.py       (约 3 分钟出图)
"""
import gym, torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from collections import deque
import random

# === 网络 ===
class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, act_dim))
    def forward(self, x): return self.net(x)

# === 经验池 ===
class ReplayBuffer:
    def __init__(self, size=50000):
        self.buf = deque(maxlen=size)
    def add(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))
    def sample(self, n=64):
        batch = random.sample(self.buf, min(n, len(self.buf)))
        return [torch.tensor(np.array(x), dtype=torch.float32) for x in zip(*batch)]
    def __len__(self): return len(self.buf)

# === Agent ===
class DQNAgent:
    def __init__(self, obs_dim, act_dim, double=True):
        self.double = double
        self.q = QNet(obs_dim, act_dim)
        self.target = QNet(obs_dim, act_dim)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=1e-3)
        self.buf = ReplayBuffer()
        self.act_dim = act_dim
        self.gamma = 0.99
        self.eps = 1.0
        self.step = 0

    def act(self, obs, eval=False):
        if not eval and random.random() < self.eps:
            return random.randrange(self.act_dim)
        with torch.no_grad():
            return self.q(torch.tensor(obs, dtype=torch.float32)).argmax().item()

    def update(self):
        if len(self.buf) < 64: return
        s, a, r, ns, d = self.buf.sample(64)
        a = a.long().unsqueeze(1)
        q_val = self.q(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            if self.double:
                best_a = self.q(ns).argmax(1, keepdim=True)
                q_next = self.target(ns).gather(1, best_a).squeeze(1)
            else:
                q_next = self.target(ns).max(1)[0]
            target = r + self.gamma * q_next * (1 - d)
        loss = nn.MSELoss()(q_val, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.step += 1
        self.eps = max(0.01, 1.0 - self.step / 10000)
        if self.step % 200 == 0:
            self.target.load_state_dict(self.q.state_dict())

# === 测试（不探索）===
def evaluate(agent, env, episodes=5):
    rewards = []
    for _ in range(episodes):
        s, done = env.reset()[0], False
        total = 0
        while not done:
            s, r, done, _, _ = env.step(agent.act(s, eval=True))
            total += r
        rewards.append(total)
    return np.mean(rewards)

# === 训练 ===
def train(double=True, episodes=300):
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, double)
    eval_rewards = []

    for ep in range(episodes):
        s, done = env.reset()[0], False
        while not done:
            a = agent.act(s)
            ns, r, done, _, _ = env.step(a)
            agent.buf.add(s, a, r, ns, float(done))
            agent.update()
            s = ns

        if (ep + 1) % 50 == 0:
            avg = evaluate(agent, env)
            eval_rewards.append(avg)
            name = "Double DQN" if double else "DQN"
            print(f"[{name}] Ep {ep+1}: eval_reward={avg:.1f}, eps={agent.eps:.2f}")

    env.close()
    return eval_rewards

if __name__ == "__main__":
    print("DQN vs Double DQN on CartPole-v1")
    print("Training...")
    dqn_rewards = train(double=False)
    ddqn_rewards = train(double=True)

    plt.plot([50 * (i + 1) for i in range(len(dqn_rewards))], dqn_rewards, label="DQN")
    plt.plot([50 * (i + 1) for i in range(len(ddqn_rewards))], ddqn_rewards, label="Double DQN")
    plt.xlabel("Episode"); plt.ylabel("Avg Reward (5 eval episodes)")
    plt.title("DQN vs Double DQN on CartPole"); plt.legend(); plt.grid()
    plt.savefig("dqn_comparison.png")
    print("Saved dqn_comparison.png")
