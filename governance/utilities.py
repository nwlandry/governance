import random

import numpy as np


def create_decision_matrix(m):
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i):
            D[j, i] = D[i, j] = random.choice([-1, 0, 1])
    return D


def create_random_opinions(n, m):
    D = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            # D[i, j] = random.choice([-1, 0, 1])
            # D[i, j] = random.uniform(-1, 1)
            if random.uniform(0, 1) > 0.4:
                D[i, j] = random.uniform(0, 1)
            else:
                D[i, j] = random.uniform(-1, 0)
    return D


def decisions_to_array(decisions):
    n = len(decisions)
    d = np.zeros(n)
    for key, val in decisions.items():
        if val == 1:
            d[key] = 1
        else:
            d[key] = -1
    return d


def truncated_normal(mean, std, bounds):
    x = np.random.normal(mean, std)
    while x <= bounds[0] or x >= bounds[1]:
        x = np.random.normal(mean, std)
    return x


def create_polarized_opinions(n, m, d, inform, pol):
    D = np.zeros((n, m))
    for i in range(n):
        if random.uniform(0, 1) < inform:  # dumb agent
            for j in range(m):
                D[i, j] = truncated_normal(0, 0.25, [-1, 1])
        elif random.uniform(0, 1) < pol:
            for j in range(m):
                D[i, j] = truncated_normal(0.5 * d[j], 0.25, [-1, 1])
        else:
            for j in range(m):
                D[i, j] = truncated_normal(-0.5 * d[j], 0.25, [-1, 1])
    return D


def create_mixed_opinions(n, m, d, inform, pol):
    D = np.zeros((n, m))
    for i in range(n):
        if random.uniform(0, 1) > inform:  # dumb agent
            for j in range(m):
                D[i, j] = truncated_normal(0, 0.25, [-1, 1])
        elif random.uniform(0, 1) < pol:
            for j in range(m):
                D[i, j] = truncated_normal(d[j], 0.25, [-1, 1])
        else:
            for j in range(m):
                D[i, j] = truncated_normal(1 - d[j], 0.25, [-1, 1])
    return D
