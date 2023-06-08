import random

import numpy as np
import xgi


# optional parameters, which go in each group. Minimal number of parameters
def decision_process(
    initial_opinions,
    decision_matrix,
    group_size,
    group_overlap,
    select_decision_type="snowball",
    select_group_type="star",
    make_decision_type="star",
    update_opinions_type="star",
):

    m = np.size(decision_matrix, axis=0)

    opinions = initial_opinions.copy()

    if np.size(decision_matrix, 0) != np.size(decision_matrix, 1):
        raise Exception("Decision matrix must be square!")

    if np.size(opinions, axis=1) != m:
        raise Exception("Opinion dimension doesn't match number of decisions")

    n = np.size(opinions, axis=0)
    nodes = np.arange(n, dtype=int)

    completed_decisions = dict()
    decisions = list(range(m))

    groups = xgi.Hypergraph()
    while len(completed_decisions) < m:
        # Select the decision to make
        cd = select_decision(
            decisions,
            completed_decisions,
            decision_matrix,
            opinions,
            how=select_decision_type,
        )

        # Select the group to make that decision
        g = select_group(
            nodes,
            group_size,
            group_overlap,
            cd,
            decision_matrix,
            opinions,
            groups,
            how=select_group_type,
        )

        groups.add_edge(g, id=cd)  # add the group to the list of all decision groups

        # Make the decision
        completed_decisions[cd] = make_decision(
            cd, g, decision_matrix, opinions, how=make_decision_type
        )
        d = completed_decisions[cd]

        # Update the group's opinions after the decision has been made.
        opinions = update_opinions(
            opinions, g, d, cd, decision_matrix, how=update_opinions_type
        )

    return completed_decisions, opinions, groups


def select_decision(
    decisions, completed_decisions, decision_matrix, opinions, how="random"
):
    """An algorithm for choosing which decision to make.

    Parameters
    ----------
    decisions : list
        The list of all decisions.
    completed_decisions : list
        The list of decisions that have already been made.
    decision_matrix : 2D numpy array
        A symmetric matrix with -1, 0, and 1 elements.
        -1  indicates a contradictory relationship between decisions,
        0 indicates no relationship between decisions, and
        1 indicates a positive relationship between decisions.
    opinions : 2D numpy array
        An array where the row indicates the decision maker, and
        the column indicates the decision. The $ij$th entry is between -1 and 1.
        It indicates the opinion of decision-maker $i$ about decision $j$;
        -1 is fully against and 1 is fully for the decision.
    how : str, default: "random"
        The method for choosing the decision to make. Options are

        - "random"
        - "sentiment"
        - "degree"

    Returns
    -------
    int
        the decision to be made
    """
    unmade_decisions = list(set(decisions).difference(set(completed_decisions.keys())))

    # This randomly chooses from the list of decisions left to be made.
    if how == "random":
        return random.choice(unmade_decisions)

    # This randomly selects the decision with probability according to how strongly
    # the population of decision makers feels.
    elif how == "sentiment":
        xavg = np.mean(np.abs(opinions), axis=0)
        xavg = xavg[unmade_decisions] / np.sum(xavg[unmade_decisions])

        return np.random.choice(unmade_decisions, size=None, replace=True, p=xavg)

    # This randomly selects the decision proportional to the number of decisions
    # (positive or negative) to which it's connected.
    elif how == "degree":
        d = np.sum(np.abs(decision_matrix), axis=0)
        d = d[unmade_decisions] / np.sum(d[unmade_decisions])
        return np.random.choice(unmade_decisions, size=None, replace=True, p=d)

    elif how == "snowball":
        pool = set()
        for dec in completed_decisions:
            neigh_dec = set(np.where(decision_matrix[dec]!=0)[0])
            pool.update(neigh_dec)

    else:
        raise Exception("Invalid decision type!")


def select_group(
    nodes, size, overlap, decision, decision_matrix, opinions, groups, how="star", **args
):
    """The algorithm for choosing the policy makers to decide on the
    selected decision.

    Parameters
    ----------
    nodes : list
        The list of all possible policy makers
    size: int
        group size
    overlap: int
        number of bridge nodes
    decision : int
        The specific decision selected on which to decide.
    decision_matrix : 2D numpy array
        A symmetric matrix with -1, 0, and 1 elements.
        -1  indicates a contradictory relationship between decisions,
        0 indicates no relationship between decisions, and
        1 indicates a positive relationship between decisions.
    opinions : 2D numpy array
        An array where the row indicates the decision maker, and
        the column indicates the decision. The $ij$th entry is between -1 and 1.
        It indicates the opinion of decision-maker $i$ about decision $j$;
        -1 is fully against and 1 is fully for the decision.
    groups : xgi.Hypergraph
        A hypergraph tabulating all the past decision makers and decisions.
    how : str, default: "random"
        The method for choosing the policy makers for the selected decision. Options are
        - "random"
        - "sentiment"
        - "degree"

    Returns
    -------
    set
        the policy makers to decide on the selected decision.
    """
    if how == "random":
        nodes = groups.nodes
        n = np.size(opinions, axis=0)
        new_nodes = set(range(n)).difference(nodes)
        # non-overlapping nodes: I had to take care of edge cases
        # (1) where there are no policy makers to begin with and
        # (2) where no policy makers have been absent from all decisions
        num_old_nodes = min(overlap, len(nodes))
        # assuming new_nodes is always a big enough population
        num_new_nodes = min(size - num_old_nodes, len(new_nodes))
        g1 = random.sample(list(new_nodes), num_new_nodes)
        g2 = random.sample(list(nodes), num_old_nodes)
        return set(g1).union(g2)
    elif how == "star":
        neigh_dec = np.where(decision_matrix[decision]!=0)[0].tolist()
        edges = neigh_dec & groups.edges
        nodes = set()
        for group in groups.edges(edges).members():
            nodes.update(group)
        n = np.size(opinions, axis=0)
        # assuming new_nodes is always a big enough population
        new_nodes = set(range(n)).difference(nodes)
        num_old_nodes = min(overlap, len(nodes))
        num_new_nodes = min(size - num_old_nodes, len(new_nodes))
        g1 = random.sample(list(new_nodes), num_new_nodes)
        g2 = random.sample(list(nodes), num_old_nodes)
        return set(g1).union(g2)
    else:
        raise Exception("Invalid group selection type!")


def make_decision(cd, decision_group, decision_matrix, opinions, how="average"):
    # average opinions
    g = sorted(decision_group)
    if how == "average":
        return np.sign(np.sum(opinions[g, cd]))

    if how == "star":
        cost_function = []
        avg_opinions = np.mean(opinions[g], axis=0)
        idx = np.where(decision_matrix[cd] != 0)
        possible_decisions = [-1, 1]
        for d in possible_decisions:
            ds = decision_matrix[cd] * d
            cost_function.append(np.sum(np.abs(ds[idx] - avg_opinions[idx])))

        i = np.argmin(cost_function)

        if len(i) > 1:
            return possible_decisions[random.choice(i)]

        return possible_decisions[i]  # wow
    else:
        raise Exception("Invalid decision making type!")


def update_opinions(opinions, decision_group, d, cd, decision_matrix, how="average"):
    """
    parameters
    ==========
    how : str, default: "average"
     - "average"
     - "sat"
    """
    g = sorted(decision_group)
    if how == "average":
        opinions[g, :] = np.mean(opinions[g, :])
    elif how == "star":
        ds = decision_matrix[cd] * d
        idx = np.where(ds!=0)
        opinions[g,idx] = ds[idx]
    else:
        raise Exception("Invalid opinion update type!")
    return opinions
