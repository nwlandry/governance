import numpy as np
import xgi

def decision_process(opinions, decisions, select_decision, select_group, make_decision, update_opinions):
    
    num_decisions = np.size(decisions, axis=0)

    if set(np.shape(decisions)) != 1:
        raise Exception("Decision matrix must be square!")

    if np.size(opinions, axis=1) != num_decisions:
        raise Exception("Opinion dimension doesn't match number of decisions")
    
    num_people = np.size(opinions, axis=0)
    
    completed_decisions = dict()

    decision_groups = xgi.Hypergraph()
    while len(completed_decisions) < num_decisions:
        # Select the decision to make
        d = select_decision(decisions, completed_decisions)

        # Select the group to make that decision
        g = select_group(decision_groups, opinions)
        decision_groups.add_edge(g, id=d) # add the group to the list of all decision groups

        # Make the decision
        completed_decisions[d] = make_decision(d, g, decisions, opinions, completed_decisions)

        # Update the group's opinions after the decision has been made.
        opinions = update_opinions(d, g, decisions, opinions, completed_decisions)

    return completed_decisions, opinions