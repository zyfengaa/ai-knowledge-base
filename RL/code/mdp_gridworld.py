"""
Grid World Value Iteration / Policy Iteration
4x4 网格世界上演示 MDP 的动态规划求解
"""
import numpy as np

# === 4x4 Grid World ===
# 状态 0-14 是可移动的，状态 15 是目标（终点）
# 动作: 0=上, 1=下, 2=左, 3=右
N_STATES = 16
ACTIONS = [0, 1, 2, 3]
GOAL = 15
# 每步奖励 -1，到达目标 0
REWARDS = np.full(N_STATES, -1.0)
REWARDS[GOAL] = 0.0

def next_state(s, a):
    """根据当前状态和动作计算下一个状态"""
    row, col = divmod(s, 4)
    if a == 0: row = max(0, row - 1)     # 上
    elif a == 1: row = min(3, row + 1)    # 下
    elif a == 2: col = max(0, col - 1)     # 左
    elif a == 3: col = min(3, col + 1)     # 右
    ns = row * 4 + col
    return s if s == GOAL else ns          # 到达终点后不动

def value_iteration(gamma=0.9, theta=1e-6):
    """值迭代算法"""
    V = np.zeros(N_STATES)
    while True:
        delta = 0
        for s in range(N_STATES):
            if s == GOAL: continue
            q_vals = []
            for a in ACTIONS:
                ns = next_state(s, a)
                q_vals.append(REWARDS[s] + gamma * V[ns])
            new_v = max(q_vals)
            delta = max(delta, abs(new_v - V[s]))
            V[s] = new_v
        if delta < theta:
            break
    return V

def policy_iteration(gamma=0.9):
    """策略迭代算法"""
    policy = np.zeros(N_STATES, dtype=int)
    V = np.zeros(N_STATES)
    while True:
        # 策略评估
        while True:
            delta = 0
            for s in range(N_STATES):
                if s == GOAL: continue
                a = policy[s]
                ns = next_state(s, a)
                new_v = REWARDS[s] + gamma * V[ns]
                delta = max(delta, abs(new_v - V[s]))
                V[s] = new_v
            if delta < 1e-6:
                break
        # 策略改进
        stable = True
        for s in range(N_STATES):
            if s == GOAL: continue
            old_a = policy[s]
            q_vals = [REWARDS[s] + gamma * V[next_state(s, a)] for a in ACTIONS]
            policy[s] = int(np.argmax(q_vals))
            if old_a != policy[s]:
                stable = False
        if stable:
            break
    return V, policy

if __name__ == "__main__":
    print("=== Value Iteration ===")
    V_vi = value_iteration()
    print("Optimal Values:")
    print(V_vi.reshape(4, 4).round(2))

    print("\n=== Policy Iteration ===")
    V_pi, policy = policy_iteration()
    action_map = {0: "上", 1: "下", 2: "左", 3: "右"}
    print("Policy:", [[action_map[a] for a in policy.reshape(4, 4)[r]] for r in range(4)])
