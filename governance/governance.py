import random

import numpy as np
import xgi


# optional parameters, which go in each group. Minimal number of parameters
def governance_process(
    initial_opinions,
    decision_matrix,
    group_size,
    group_overlap,
    select_decision_type="snowball",
    select_group_type="star",
    make_decision_type="star",
    update_opinions_type="star",
):
    """This implements a governance process where
    decisions are made by stakeholders.

    Parameters
    ----------
    initial_opinions : numpy ndarray
        an N x D matrix, where N is the number of nodes
        and D is the number of decisions to be made
    decision_matrix : numpy ndarray
        a D x D matrix encoding the negative and
        positive relationships (AND and XOR) between
        decisions.
    group_size : int >= 2
        The size of the stakeholder groups that are making
        the decisions
    group_overlap : int <= group_size
        The number of past decision makers added to a new
        group of stakeholders
    select_decision_type : str, optional
        The way in which a policy issue issue to vote on is
        chosen, by default "snowball". For more information, see `select_decision`.
    select_group_type : str, optional
        The way in which a group of stakeholders is chosen, by default "star".
        For more information, see `select_group`.
    make_decision_type : str, optional
        The way in which a group of stakeholders decides on a policy issue,
        by default "star". For more information, see `make_decision`.
    update_opinions_type : str, optional
        The way in which the nodes update their opinions about all the
        policy issues, by default "star". For more information, see `update_opinion`.

    Returns
    -------
    completed_decisions, opinions, groups : (dict, numpy array, xgi.Hypergraph)
        `completed_decisions` is a dictionary where the keys are the decision
        index and the values are -1/1, indicating voting against/for.
        `opinions` is a numpy array with the updated opinions of all the nodes.
        `group` is an xgi Hypergraph indicating the nodes involved in each decision
        and the hyperedge IDs indicate the decision IDs.

    Raises
    ------
    Exception
        If the decision matrix isn't square or the number of opinions for each
        node doesn't match the number of decisions.
    """
    opinions = initial_opinions.copy()
    n, d = opinions.shape
    d1, d2 = decision_matrix.shape

    if d1 != d2:
        raise Exception("The decision matrix must be square.")

    if d != d1:
        raise Exception(
            "The number of opinions for each node doesn't match the number of decisions."
        )

    if group_overlap > group_size:
        raise Exception("The group size must be equal to or larger than the overlap.")

    nodes = np.arange(n, dtype=int)

    completed_decisions = dict()
    decisions = list(range(d))

    groups = xgi.Hypergraph()
    while len(completed_decisions) < d:
        # Select the decision to make
        decision = select_decision(
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
            decision,
            decision_matrix,
            opinions,
            groups,
            how=select_group_type,
        )

        groups.add_edge(
            g, id=decision
        )  # add the group to the list of all decision groups

        # Make the decision
        choice = make_decision(
            decision, g, decision_matrix, opinions, how=make_decision_type
        )
        completed_decisions[decision]
        # Update the group's opinions after the decision has been made.
        opinions = update_opinions(
            opinions,
            g,
            choice,
            decision,
            decision_matrix,
            how=update_opinions_type,
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

        - "random": choose the policy issue uniformly at random
        - "sentiment": choose the policy issue at random proportional to the
        the absolute value of the sentiment averaged over the population.
        - "degree": choose the policy issue proportional to the number of
        other policy issues to which it is connected.
        - "snowball": choose the policy issues uniformly at random from a list of
        decisions adjacent to all the decisions that have already been made.

    Returns
    -------
    int
        the decision to be made

    Raises
    ------
    Exception
        if an invalid decision type is made
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
        if len(completed_decisions) == 0:
            return random.choice(list(unmade_decisions))

        pool = set()
        for dec in completed_decisions:
            neigh_dec = set(np.where(decision_matrix[dec] != 0)[0].tolist())
            pool.update(neigh_dec)

        possible_decisions = list(pool.intersection(unmade_decisions))
        return random.choice(possible_decisions)

    else:
        raise Exception("Invalid decision type!")


def select_group(
    size, overlap, decision, decision_matrix, opinions, groups, how="star", **args
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

        - "random": Choose both the new policy makers uniformly at random and choose
        the overlapping policy makers uniformly at random from the list of prior decision
        makers
        - "star": Choose new policy makers uniformly at random and overlapping policy makers
        uniformly at random from the list of decision makers who have decided on policies affected
        by/affecting this policy.

    Returns
    -------
    set
        the policy makers to decide on the selected decision.
    """
    if how == "random":
        old_nodes = groups.nodes
        n = np.size(opinions, axis=0)
        new_nodes = set(range(n)).difference(old_nodes)

        # edge cases:
        # (1) where there are no policy makers to begin with and
        # (2) where no policy makers have been absent from all decisions
        num_old_nodes = min(overlap, len(old_nodes))
        num_new_nodes = min(size - num_old_nodes, len(new_nodes))

        g1 = random.sample(list(new_nodes), num_new_nodes)
        g2 = random.sample(list(old_nodes), num_old_nodes)
        return set(g1).union(g2)
    elif how == "star":
        neigh_dec = np.where(decision_matrix[decision] != 0)[0].tolist()

        # completed decisions affected by/affecting this one
        edges = neigh_dec & groups.edges
        old_nodes = set()

        # get all the policy makers from those decisions
        for group in groups.edges(edges).members():
            old_nodes.update(group)
        n = np.size(opinions, axis=0)

        # handling the edge cases as before
        new_nodes = set(range(n)).difference(old_nodes)
        num_old_nodes = min(overlap, len(old_nodes))
        num_new_nodes = min(size - num_old_nodes, len(new_nodes))

        g1 = random.sample(list(new_nodes), num_new_nodes)
        g2 = random.sample(list(old_nodes), num_old_nodes)
        return set(g1).union(g2)
    else:
        raise Exception("Invalid group selection type!")


def make_decision(decision, decision_group, decision_matrix, opinions, how="average"):
    """The method by which a group of policy makers votes for/against a policy.

    Parameters
    ----------
    decision : int
        The decision index
    decision_group : set
        The list of policy makers making the decision
    decision_matrix : numpy ndarray
        A D x D matrix encoding the relationships between decisions.
    opinions : numpy nd arrray
        An N x D matrix of the opinions that each policy maker holds
        on each of the policies.
    how : str, optional
        The method by which the decision is made, by default "average".
        Current choices are:

        - "average": returns the sign of the average group sentiment
        - "star": minimizes the average dissatisfaction across the decision
        and coherent related decisions.

    Returns
    -------
    int
        -1 if voting against the policy, 1 if voting for the policy.

    Raises
    ------
    Exception
        If invalid decision making type is chosen.
    """
    # average opinions
    g = sorted(decision_group)
    if how == "average":
        return np.sign(opinions[g, decision].sum())

    if how == "star":
        cost_function = []
        # calculate the average opinion of the group.
        avg_opinions = opinions[g].mean(axis=0)

        # find related decisions
        idx = np.where(decision_matrix[decision] != 0)
        possible_decisions = [-1, 1]
        for d in possible_decisions:
            ds = decision_matrix[decision] * d

            # calculate the dissatisfaction across the decision and all
            # (coherent) related decisions.
            cost_function.append(np.sum(np.abs(ds[idx] - avg_opinions[idx])))

        # minimize the dissatisfaction
        i = np.argmin(cost_function)

        return possible_decisions[i]
    else:
        raise Exception("Invalid decision making type!")


def update_opinions(
    opinions, decision_group, choice, decision, decision_matrix, how="average"
):
    """The method by which stakeholders update their opinions after making a decision

    Parameters
    ----------
    opinions : numpy nd arrray
        An N x D matrix of the opinions that each policy maker holds
        on each of the policies.
    decision_group : set
        The list of policy makers making the decision
    choice : int
        -1 if decided against the policy, `1 if decided for it
    decision : int
        the index of the decision
    decision_matrix : numpy ndarray
        A D x D matrix encoding the relationship between decisions.
    how : str, optional
        The method by which the nodes update their opinions, by default "average"

    Returns
    -------
    numpy ndarray
        An N x D matrix with the updated opinions of all nodes on all policy
        decisions.

    Raises
    ------
    Exception
        If an invalid updating method is described.
    """
    g = sorted(decision_group)
    if how == "average":
        opinions[g, :] = np.mean(opinions[g, :])

    elif how == "star":
        ds = decision_matrix[decision] * choice
        idx = np.where(ds != 0)
        for i in g:
            # update the opinions in the group of stakeholders
            # to match the decision made and be coherent with the
            # related decisions as well.
            opinions[i, idx] = ds[decision]
    else:
        raise Exception("Invalid opinion update type!")
    return opinions
