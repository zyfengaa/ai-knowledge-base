# rlhf_reward_demo.py -- RLHF core: train reward model from preferences
# Simulates phase 1 of RLHF: learn a reward model from preference comparisons.
# Bradley-Terry loss: L = -log(sigmoid(r(y_w) - r(y_l)))
import random, math
random.seed(42)

# ====== hidden true scoring function (unknown to reward model) ======
TRUE_W = [0.40, 0.25, 0.15, 0.12, 0.08]

def true_score(f):
    return sum(w * v for w, v in zip(TRUE_W, f))

def gen_pair():
    a = [random.random() for _ in range(5)]
    b = [random.random() for _ in range(5)]
    label = 1 if true_score(a) >= true_score(b) else 0
    return a, b, label

# ====== linear reward model: r(x) = w*x + b ======
class RewardModel:
    def __init__(self, dim=5, lr=0.01):
        self.w = [0.0] * dim
        self.b = 0.0
        self.lr = lr
    def score(self, x):
        return sum(w * v for w, v in zip(self.w, x)) + self.b
    def sigmoid(self, x):
        if x > 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)
    def update(self, a, b, label):
        # Bradley-Terry loss: L = -log(sigmoid(r(y_w) - r(y_l)))
        sa, sb = self.score(a), self.score(b)
        delta = sa - sb
        sig = self.sigmoid(delta)
        # gradient of Bradley-Terry loss
        if label == 1:   # a preferred: minimize -log(sig(delta))
            dsa, dsb = sig - 1.0, 1.0 - sig
        else:            # b preferred: minimize -log(sig(-delta))
            dsa, dsb = sig, -sig
        prob = sig if label == 1 else (1.0 - sig)
        loss = -math.log(max(prob, 1e-10))
        for i in range(5):
            self.w[i] -= self.lr * (dsa * a[i] + dsb * b[i])
        self.b -= self.lr * (dsa + dsb)
        return loss

# ====== training ======
rm = RewardModel(dim=5, lr=0.05)
data = [gen_pair() for _ in range(3000)]
valid = [gen_pair() for _ in range(500)]
print("=" * 55)
print("  RLHF Reward Model Training")
print("  3000 train + 500 validation pairs")
print("=" * 55)
print()
print("  True weights: {}".format(TRUE_W))
print()

for epoch in range(50):
    random.shuffle(data)
    total_loss = 0.0
    for a, b, label in data:
        total_loss += rm.update(a, b, label)
    correct = 0
    for a, b, label in valid:
        pred = 1 if rm.score(a) >= rm.score(b) else 0
        correct += (pred == label)
    if epoch % 10 == 0:
        print("  epoch {:2d}:  loss={:.4f}  val_acc={:.3f}".format(epoch, total_loss / len(data), correct / len(valid)))

# ====== evaluation ======
print()
print("--- Learned weights vs True weights ---")
wn = math.sqrt(sum(w * w for w in rm.w))
for i in range(5):
    ratio = rm.w[i] / wn if wn > 0 else 0.0
    print("  dim{:d}:  true={:.2f}  learned_dir={:+.2f}".format(i, TRUE_W[i], ratio))

test = [[random.random() for _ in range(5)] for _ in range(50)]
true_order = sorted(range(50), key=lambda i: true_score(test[i]), reverse=True)
model_order = sorted(range(50), key=lambda i: rm.score(test[i]), reverse=True)
agree = 0
for i in range(50):
    for j in range(i + 1, 50):
        tr = 1 if true_order.index(i) < true_order.index(j) else 0
        mr = 1 if model_order.index(i) < model_order.index(j) else 0
        agree += 1 if tr == mr else 0
total = 50 * 49 // 2
print()
print("--- Ranking consistency (50 items, {} pairs) ---".format(total))
print("  Consistent: {} / {} ({:.1%})".format(agree, total, agree / total))
print()
print("Conclusion: Reward model learned the hidden preference structure.")
print("This is phase 1 of RLHF (training reward model from preferences).")