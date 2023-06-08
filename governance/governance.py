import random

import numpy as np
import xgi


# optional parameters, which go in each group. Minimal number of parameters
def decision_process(
    opinions,
    decision_network,
    group_size,
    group_overlap,
    select_decision_type="random",
    select_group_type="random",
    make_decision_type="average",
    update_opinions_type="average",
):

    m = np.size(decision_network, axis=0)

    o = opinions.copy()

    if np.size(decision_network, 0) != np.size(decision_network, 1):
        raise Exception("Decision matrix must be square!")

    if np.size(o, axis=1) != m:
        raise Exception("Opinion dimension doesn't match number of decisions")

    n = np.size(o, axis=0)
    nodes = np.arange(n, dtype=int)

    completed_decisions = dict()
    decisions = list(range(m))

    groups = xgi.Hypergraph()
    while len(completed_decisions) < m:
        # Select the decision to make
        d = select_decision(
            decisions, completed_decisions, decision_network, o, how=select_decision_type
        )

        # Select the group to make that decision
        g = select_group(
            nodes,
            group_size,
            group_overlap,
            d,
            decision_network,
            o,
            groups,
            how=select_group_type,
        )

        groups.add_edge(
            g, id=d
        )  # add the group to the list of all decision groups

        # Make the decision
        completed_decisions[d] = make_decision(
            d, g, decision_network, o, completed_decisions, how=make_decision_type
        )

        # Update the group's opinions after the decision has been made.
        o = update_opinions(o, g, decision_network, how=update_opinions_type)

    return completed_decisions, o, groups


def select_decision(
    decisions, completed_decisions, decision_network, opinions, how="random"
):
    """An algorithm for choosing which decision to make.

    Parameters
    ----------
    decisions : list
        The list of all decisions.
    completed_decisions : list
        The list of decisions that have already been made.
    decision_network : 2D numpy array
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
        d = np.sum(np.abs(decision_network), axis=0)
        d = d[unmade_decisions] / np.sum(d[unmade_decisions])
        return np.random.choice(unmade_decisions, size=None, replace=True, p=d)
    else:
        raise Exception("Invalid decision type!")


def select_group(
    nodes,
    size,
    overlap,
    decision,
    decision_network,
    opinions,
    groups,
    how="random",
    **args
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
    decision_network : 2D numpy array
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
    n = np.size(opinions, axis=0)
    nodes = set(groups.nodes)
    new_nodes = set(range(n)).difference(nodes)
    if "random":
        # non-overlapping nodes: I had to take care of edge cases
        # (1) where there are no policy makers to begin with and
        # (2) where no policy makers have been absent from all decisions
        new_nodes = min(size - overlap, len(new_nodes))
        old_nodes = min(overlap, len(nodes))
        g1 = random.sample(list(new_nodes), new_nodes)
        g2 = random.sample(list(nodes), old_nodes)
        return set(g1).union(g2)
    else:
        raise Exception("Invalid group selection type!")


def make_decision(
    d, decision_group, decision_network, opinions, completed_decisions, how="average"
):
    # average opinions
    if how == "average":
        return np.mean(opinions[list(decision_group), d]) > 0
    if how == "star":
        config1 = -1
        return config1
    else:
        raise Exception("Invalid decision making type!")


def update_opinions(opinions, decision_group, d, decision_network, how="average"):
    g = sorted(decision_group)
    if how == "average":
        opinions[g, :] = np.mean(opinions[g, :])
    elif how == "sat":
        
        opinions[g, :] = np.mean(opinions[g, :])
    else:
        raise Exception("Invalid opinion update type!")
    return opinions
