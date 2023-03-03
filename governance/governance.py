import numpy as np
import xgi


# optional parameters, which go in each group. Minimal number of parameters
def decision_process(
    opinions,
    decision_matrix,
    select_decision,
    select_group,
    make_decision,
    update_opinions,
    group_size,
    group_overlap,
):

    num_decisions = np.size(decision_matrix, axis=0)

    if np.size(decision_matrix, 0) != np.size(decision_matrix, 1):
        raise Exception("Decision matrix must be square!")

    if np.size(opinions, axis=1) != num_decisions:
        raise Exception("Opinion dimension doesn't match number of decisions")

    num_policy_makers = np.size(opinions, axis=0)
    policy_makers = list(range(num_policy_makers))

    completed_decisions = dict()
    decisions = list(range(num_decisions))

    decision_groups = xgi.Hypergraph()
    while len(completed_decisions) < num_decisions:
        # Select the decision to make
        d = select_decision(decisions, completed_decisions, decision_matrix, opinions)

        # Select the group to make that decision
        g = select_group(
            policy_makers,
            group_size,
            group_overlap,
            d,
            decision_matrix,
            opinions,
            decision_groups,
            how="random",
        )
        decision_groups.add_edge(
            g, id=d
        )  # add the group to the list of all decision groups

        # Make the decision
        completed_decisions[d] = make_decision(
            d, g, decision_matrix, opinions, completed_decisions
        )

        # Update the group's opinions after the decision has been made.
        opinions = update_opinions(opinions, g, decision_matrix)

    return completed_decisions, opinions
