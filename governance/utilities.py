import math
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
            D[i, j] = random.uniform(-1, 1)
            # if random.uniform(0, 1) > 0.4:
            #    D[i, j] = random.uniform(0, 1)
            # else:
            #    D[i, j] = random.uniform(-1, 0)
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
                D[i, j] = truncated_normal(0, 0.1, [-1, 1])
        elif random.uniform(0, 1) < pol:
            for j in range(m):
                D[i, j] = truncated_normal(0.5 * d[j], 0.1, [-1, 1])
        else:
            for j in range(m):
                D[i, j] = truncated_normal(-0.5 * d[j], 0.1, [-1, 1])
    return D


def create_mixed_opinions(n, m, d, inform, pol):
    D = np.zeros((n, m))
    maximum = 1.0
    for j in range(m):
        if d[j] > maximum:
            maximum = d[j]
        elif (1 - d[j]) > maximum:
            maximum = 1 - d[j]

    for i in range(n):
        if random.uniform(0, 1) < inform:  # dumb agent
            for j in range(m):
                D[i, j] = truncated_normal(0, 0.1, [-1, 1])
        elif random.uniform(0, 1) < pol:
            for j in range(m):
                D[i, j] = truncated_normal(d[j] / maximum, 0.1, [-1, 1])
        else:
            for j in range(m):
                D[i, j] = truncated_normal((1 - d[j]) / maximum, 0.1, [-1, 1])
    return D


def create_incoherent_opinions(n, m, d):
    D = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            D[i, j] = truncated_normal(d[j], 0.1, [-1, 1])
    return D


def create_greedy_opinions(n, m, G):
    D = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            D[i, j] = truncated_normal(0.0, 0.1, [-1, 1])
        completed_decisions = []
        unmade_decisions = list(
            set(np.arange(0, m)).difference(set(completed_decisions))
        )
        seed = random.choice(unmade_decisions)
        D[i, seed] = truncated_normal(0.5, 0.1, [-1, 1])
        completed_decisions.append(seed)
        pool = set()
        unmade_decisions = list(
            set(np.arange(0, m)).difference(set(completed_decisions))
        )
        while len(unmade_decisions) > 0:
            for dec in completed_decisions:
                neigh_dec = set(np.where(G[dec] != 0)[0].tolist())
                pool.update(neigh_dec)
                possible_decisions = list(pool.intersection(unmade_decisions))
                previous_dec = math.copysign(1, D[i, dec])
                if len(unmade_decisions) == 0:
                    break
                if len(possible_decisions) == 0:
                    seed = random.choice(unmade_decisions)
                    D[i, seed] = truncated_normal(0.5, 0.1, [-1, 1])
                    completed_decisions.append(seed)
                    unmade_decisions = list(
                        set(np.arange(0, m)).difference(set(completed_decisions))
                    )
                    if len(unmade_decisions) == 0:
                        break
                for next_dec in possible_decisions:
                    D[i, next_dec] = truncated_normal(
                        previous_dec * G[dec, next_dec] * 0.5, 0.25, [-1, 1]
                    )
                    completed_decisions.append(next_dec)
                    unmade_decisions = list(
                        set(np.arange(0, m)).difference(set(completed_decisions))
                    )
                    if len(unmade_decisions) == 0:
                        break

    return D


def create_uniform_greedy_opinions(n, m, G):
    D = np.zeros((n, m))
    completed_decisions = []
    unmade_decisions = list(set(np.arange(0, m)).difference(set(completed_decisions)))
    seed = random.choice(unmade_decisions)
    for i in range(n):
        D[i, seed] = truncated_normal(0.75, 0.1, [-1, 1])
    completed_decisions.append(seed)
    pool = set()
    unmade_decisions = list(set(np.arange(0, m)).difference(set(completed_decisions)))
    while len(unmade_decisions) > 0:
        for dec in completed_decisions:
            neigh_dec = set(np.where(G[dec] != 0)[0].tolist())
            pool.update(neigh_dec)
            possible_decisions = list(pool.intersection(unmade_decisions))
            sensing = 0
            for i in range(n):
                sensing += D[i, dec]
            previous_dec = math.copysign(1, sensing)
            if len(unmade_decisions) == 0:
                break
            if len(possible_decisions) == 0:
                seed = random.choice(unmade_decisions)
                for i in range(n):
                    D[i, seed] = truncated_normal(0.75, 0.1, [-1, 1])
                completed_decisions.append(seed)
                unmade_decisions = list(
                    set(np.arange(0, m)).difference(set(completed_decisions))
                )
                if len(unmade_decisions) == 0:
                    break
            for next_dec in possible_decisions:
                for i in range(n):
                    D[i, next_dec] = truncated_normal(
                        previous_dec * G[dec, next_dec] * 0.75, 0.1, [-1, 1]
                    )
                completed_decisions.append(next_dec)
                unmade_decisions = list(
                    set(np.arange(0, m)).difference(set(completed_decisions))
                )
                if len(unmade_decisions) == 0:
                    break

    return D
