import random

import numpy as np


def select_decision(decisions, past_decisions, decision_matrix, opinions, how="random"):
    """An algorithm for choosing which decision to make.

    Parameters
    ----------
    decisions : list
        The list of all decisions.
    past_decisions : list
        The list of decisions that have already been made.
    decision_matrix : 2D numpy array
        A symmetric matrix with -1, 0, and 1 elements.
        -1  indicates a contradictory relationship between decisions,
        0 indicates no relationship between decisions, and
        1 indicates a positive relationship between decisions
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
    _type_
        _description_
    """
    unmade_decisions = list(set(decisions).difference(set(past_decisions)))

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


def select_group(num_people, start, stop):
    return random.sample(range(num_people), random.randrange(start, stop))


def update_opinions(opinions, decision_group, decision):
    return 0
