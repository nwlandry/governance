import random

def select_decision(decisions, past_decisions):
    unmade_decisions = list(set(decisions).difference(set(past_decisions)))
    return random.choice(unmade_decisions)

def select_group(num_people, start, stop):
    return random.sample(range(num_people), random.randrange(start, stop))

def update_opinions(opinions, decision_group, decision):
    return 0