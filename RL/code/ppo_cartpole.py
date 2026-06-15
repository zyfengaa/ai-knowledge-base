"""
PPO — CartPole-v1 完整训练 + 学习曲线
包含：Actor-Critic / Clip / GAE / 多轮更新 / 评测 / 画图

运行: pip install torch gym matplotlib
      python code/ppo_cartpole.py       (约 3 分钟出图)
"""
import gym, torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from torch.distributions import Categorical

# === 1. Actor-Critic ===
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh())
        self.policy = nn.Linear(64, act_dim)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        h = self.net(x)
        return self.policy(h), self.value(h).squeeze(-1)

    def step(self, obs):
        logits, val = self(obs)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a.item(), dist.log_prob(a), val

# === 2. 完整轨迹采样 ===
def rollout(env, model, max_steps=500):
    s, done = env.reset()[0], False
    data = {"states": [], "actions": [], "log_probs": [], "rewards": [], "values": [], "dones": []}
    while not done and len(data["rewards"]) < max_steps:
        a, lp, v = model.step(torch.tensor(s, dtype=torch.float32))
        ns, r, done, _, _ = env.step(a)
        data["states"].append(s); data["actions"].append(a)
        data["log_probs"].append(lp); data["rewards"].append(r)
        data["values"].append(v.item()); data["dones"].append(float(done))
        s = ns
    return data

# === 3. GAE 计算 ===
def calc_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv.insert(0, gae)
    ret = [v + a for v, a in zip(values, adv)]
    return torch.tensor(adv, dtype=torch.float32), torch.tensor(ret, dtype=torch.float32)

# === 4. 评估 ===
def evaluate(env, model, episodes=5):
    total = 0
    for _ in range(episodes):
        s, done = env.reset()[0], False
        ep_r = 0
        while not done:
            with torch.no_grad():
                logits, _ = model(torch.tensor(s, dtype=torch.float32))
                a = Categorical(logits=logits).sample().item()
            s, r, done, _, _ = env.step(a)
            ep_r += r
        total += ep_r
    return total / episodes

# === 5. 训练 ===
def train(episodes=300):
    env = gym.make("CartPole-v1")
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    eval_rewards = []

    for ep in range(episodes):
        # 采样轨迹
        data = rollout(env, model)
        adv, ret = calc_gae(data["rewards"], data["values"], data["dones"])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        s = torch.tensor(np.array(data["states"]), dtype=torch.float32)
        a = torch.tensor(data["actions"])
        old_lp = torch.stack(data["log_probs"])

        # PPO 多轮更新
        for _ in range(4):
            logits, vals = model(s)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(a)
            ratio = torch.exp(new_lp - old_lp.detach())
            clipped = torch.clamp(ratio, 0.8, 1.2)
            loss_p = -torch.min(ratio * adv, clipped * adv).mean()
            loss_v = nn.MSELoss()(vals, ret)
            loss = loss_p + 0.5 * loss_v

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

        # 定期评测
        if (ep + 1) % 50 == 0:
            avg = evaluate(env, model)
            eval_rewards.append(avg)
            print(f"[PPO] Ep {ep+1}: eval_reward={avg:.1f}")

    env.close()
    return eval_rewards

if __name__ == "__main__":
    print("PPO on CartPole-v1")
    rewards = train()
    plt.plot([50 * (i+1) for i in range(len(rewards))], rewards, "-o")
    plt.xlabel("Episode"); plt.ylabel("Avg Reward (5 test episodes)")
    plt.title("PPO on CartPole-v1"); plt.grid(); plt.savefig("ppo_result.png")
    print("Saved ppo_result.png")
