import math
import matplotlib.pyplot as plt

# Taken by https://ai.stackexchange.com/questions/39896/choosing-and-designing-decay-types-for-epsilon-greedy-exploration-in-reinforceme
def exponential_epsilon_decay(step_idx, epsilon_start=1, epsilon_end=0.01, epsilon_decay=100_000):
    """
    Calculates the value of epsilon for a given step index using exponential decay and the specified parameters.

    Parameters:
    step_idx (int): The index of the current step.
    epsilon_start (float): The starting value of epsilon.
    epsilon_end (float): The minimum value of epsilon.
    epsilon_decay (float): The rate at which epsilon decays.

    Returns:
    float: The value of epsilon for the given step index.
    """
    return epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * step_idx / epsilon_decay)

def linear_epsilon_decay(step_idx, epsilon_start=1, epsilon_end=0.01, epsilon_decay=100_000):
    """
    Calculates the value of epsilon for a given step index using linear decay and the specified parameters.

    Parameters:
    step_idx (int): The index of the current step.
    epsilon_start (float): The starting value of epsilon.
    epsilon_end (float): The minimum value of epsilon.
    epsilon_decay (float): The total number of steps over which epsilon will decay from epsilon_start to epsilon_end.

    Returns:
    float: The value of epsilon for the given step index.
    """
    return epsilon_end + (epsilon_start - epsilon_end) * max((1 - step_idx / epsilon_decay), 0)