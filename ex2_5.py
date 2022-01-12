import time

import numpy as np
import matplotlib.pyplot as plt


S = 10000
N = 2000
K = 10
mu = np.zeros(N)
sigma = np.ones(N) * 0.01

# Qs = np.ones((N, K)) * np.random.random()
Qs = np.random.normal(0, 1, N * K).reshape((N, K))
var = 1.0

eps1 = 0.1
eps2 = 0.01
eps3 = 0.0


def get_action(q: np.ndarray, eps: float) -> np.ndarray:
    actions = []
    for i in range(N):
        # print(q[i])
        if np.random.random() < eps:
            a = np.random.choice(K)
        else:
            a = np.argmax(q[i])
        # print(a)
        # input("just checking...")
        actions.append(a)
    return np.array(actions)


def step(a: int, n: int) -> float:
    return np.random.normal(Qs[n, a], var, 1)[0]


start = time.time()
for eps in [eps1, eps2, eps3]:
    # for eps in [eps1]:
    print(f"Running for eps: {eps}")
    st_eps = time.time()
    Q = np.random.rand(N, K)
    plays = np.zeros((N, K))
    ys = []
    xs = list(range(1, S + 1))
    st = time.time()
    for idx in range(S):
        actions = get_action(Q, eps)
        avg_r = 0
        for i, a in enumerate(actions):
            r = step(a, i)
            plays[i, a] += 1
            Q[i, a] += (1 / plays[i, a]) * (r - Q[i, a])
            avg_r += r
        avg_r /= len(actions)
        ys.append(avg_r)
        if idx % 100 == 99:
            print(f"Completed steps {idx - 98}-{idx + 1} in {time.time() - st} seconds.")
            st = time.time()
    print(f"Completed an eps in {time.time() - st_eps} seconds.")
    plt.plot(xs, ys, label=f"eps = {eps}")
plt.legend()
plt.show()
print(f"Finished all eps in {time.time() - start} seconds.")
