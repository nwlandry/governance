import random

import numpy as np


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
    if how == "sentiment":
        xavg = np.mean(np.abs(opinions), axis=0)

        return random.choice(
            unmade_decisions, size=None, replace=True, p=xavg[unmade_decisions]
        )

    # This randomly selects the decision proportional to the number of decisions
    # (positive or negative) to which it's connected.
    if how == "degree":
        d = np.sum(np.abs(decision_matrix))
        return random.choice(
            unmade_decisions, size=None, replace=True, p=d[unmade_decisions]
        )


def select_group(
    policy_makers,
    size,
    overlap,
    decision,
    decision_matrix,
    opinions,
    H,
    how="random",
    **args
):
    """The algorithm for choosing the policy makers to decide on the
    selected decision.

    Parameters
    ----------
    policy_makers : list
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
    H : xgi.Hypergraph
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
    policy_makers = set(H.nodes)
    new_policy_makers = set(range(n)).difference(policy_makers)
    if "random":
        # non-overlapping nodes: I had to take care of edge cases
        # (1) where there are no policy makers to begin with and
        # (2) where no policy makers have been absent from all decisions
        new_nodes = min(size - overlap, len(new_policy_makers))
        old_nodes = min(overlap, len(policy_makers))
        g1 = random.sample(new_policy_makers, new_nodes)
        g2 = random.sample(policy_makers, old_nodes)
        return set(g1).union(g2)


def make_decision(d, decision_group, decision_matrix, opinions, completed_decisions):
    # average opinions
    return np.mean(opinions[list(decision_group), d]) > 0


def update_opinions(opinions, decision_group, decision_matrix):
    opinions[list(decision_group), :] = np.mean(opinions[list(decision_group), :])
    return opinions
