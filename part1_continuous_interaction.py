"""Part 1: continuous-interaction agent learning object-label associations in a changing grid world."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_COLORS, N_SHAPES, N_LABELS = 4, 4, 6
N_FEAT = N_COLORS + N_SHAPES + N_COLORS * N_SHAPES
GRID = 20
N_OBJECTS = 40
STEPS = 6000
CHANGE_AT = STEPS // 2
LR, DECAY, NOISE = 0.15, 0.002, 0.15
NOVELTY = 1.0


class Environment:
    def __init__(self, rng):
        self.rng = rng
        self.pos = rng.integers(0, GRID, size=(N_OBJECTS, 2))
        self.color = rng.integers(0, N_COLORS, N_OBJECTS)
        self.shape = rng.integers(0, N_SHAPES, N_OBJECTS)
        self.rule = rng.integers(0, N_LABELS, size=(N_COLORS, N_SHAPES))

    def change(self):
        mask = self.rng.random(self.rule.shape) < 0.5
        self.rule[mask] = self.rng.integers(0, N_LABELS, mask.sum())

    def features(self, i):
        x = np.zeros(N_FEAT)
        x[self.color[i]] = 1.0
        x[N_COLORS + self.shape[i]] = 1.0
        x[N_COLORS + N_SHAPES + self.color[i] * N_SHAPES + self.shape[i]] = 1.0
        return x + self.rng.normal(0, NOISE, x.shape)

    def label(self, i):
        return self.rule[self.color[i], self.shape[i]]


class Agent:
    def __init__(self, rng, policy):
        self.rng = rng
        self.policy = policy
        self.W = np.zeros((N_FEAT, N_LABELS))
        self.xy = np.array([GRID // 2, GRID // 2])
        self.target = None
        self.last_visit = np.zeros(N_OBJECTS)
        self.t = 0

    def predict(self, x):
        z = self.W.T @ x
        p = np.exp(z - z.max())
        return p / p.sum()

    def uncertainty(self, x):
        p = self.predict(x)
        return -(p * np.log(p + 1e-12)).sum()

    def choose_target(self, env):
        if self.policy == "active":
            u = np.array([self.uncertainty(env.features(i)) for i in range(N_OBJECTS)])
            u += NOVELTY * (self.t - self.last_visit) / STEPS
            u += self.rng.normal(0, 0.05, u.shape)
            return int(np.argmax(u))
        return int(self.rng.integers(N_OBJECTS))

    def step(self, env):
        self.t += 1
        if self.target is None:
            self.target = self.choose_target(env)
        goal = env.pos[self.target]
        self.xy = self.xy + np.sign(goal - self.xy)
        if np.all(self.xy == goal):
            x = env.features(self.target)
            y = np.zeros(N_LABELS)
            y[env.label(self.target)] = 1.0
            p = self.predict(x)
            self.W += LR * np.outer(x, y - p)
            self.W *= (1 - DECAY)
            self.last_visit[self.target] = self.t
            self.target = None
            return True
        return False

    def accuracy(self, env):
        preds = [np.argmax(self.predict(env.features(i))) for i in range(N_OBJECTS)]
        return np.mean([preds[i] == env.label(i) for i in range(N_OBJECTS)])


def run(policy, seed):
    rng = np.random.default_rng(seed)
    env = Environment(rng)
    agent = Agent(rng, policy)
    acc = np.zeros(STEPS)
    for t in range(STEPS):
        if t == CHANGE_AT:
            env.change()
        agent.step(env)
        acc[t] = agent.accuracy(env)
    return acc


if __name__ == "__main__":
    seeds = range(10)
    plt.figure(figsize=(8, 4))
    for policy, color in [("active", "C0"), ("random", "C1")]:
        curves = np.array([run(policy, s) for s in seeds])
        m, sd = curves.mean(0), curves.std(0)
        plt.plot(m, color=color, label=f"{policy} exploration")
        plt.fill_between(range(STEPS), m - sd, m + sd, color=color, alpha=0.2)
        print(f"{policy:7s}: acc before change (t={CHANGE_AT-1}) = {m[CHANGE_AT-1]:.2f}, "
              f"right after = {m[CHANGE_AT+1]:.2f}, end = {m[-1]:.2f}")
    plt.axvline(CHANGE_AT, color="k", ls="--", label="environment changes")
    plt.xlabel("time step (continuous interaction)")
    plt.ylabel("association accuracy over all objects")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.title("Continuous interaction: learning, disruption, and re-adaptation")
    plt.tight_layout()
    plt.savefig("results/part1_learning_curve.png", dpi=150)
    print("saved results/part1_learning_curve.png")
