import numpy as np
import pandas as pd

class HiddenMarkovChain:
    def __init__(self, n_states:int, transition_probability_matrix:np.array, emission_probabilities: np.array, initial_state_probability_distribution:np.array):
        """
        - n_states is just a check
        - transition_probability_matrix dimension: n_states x n_states
        - emission probabilities dimension: T_observations x n_states, i.e. P(o_i|state = STATE) for i = 1, 2, ..., T
        - initial_probability_distribution: n_states x 1 
        """

        if (len(transition_probability_matrix) != n_states or len(transition_probability_matrix[0]) != n_states):
            raise Exception(f"Transition Probability matrix should be {n_states}x{n_states}")
        
        self.n_states = n_states
        self.transition_probability_matrix = transition_probability_matrix
        self.emission_probabilities = emission_probabilities
        self.initial_state_probability_distribution = initial_state_probability_distribution

    def likelihood_seq(self, observation_seq: list):
        trellis_matrix = [[0] * len(observation_seq) for _ in range(self.n_states)]
        
        # initialize
        for state in range(self.n_states):
            trellis_matrix[state][0] = self.initial_state_probability_distribution[state] * self.emission_probabilities[observation_seq[0]][state]

        for seq in range(1, len(observation_seq)):
            for state in range(self.n_states):
                for i in range(self.n_states):
                    # print(f"state: {state}, seq: {seq}, i: {i}, obs seq: {observation_seq[seq]}, trellis matrix: {trellis_matrix[i][seq-1]}, transition prob: {self.transition_probability_matrix[state][i]}, emission prob: {self.emission_probabilities[observation_seq[seq]][state]}")
                    trellis_matrix[state][seq] += trellis_matrix[i][seq-1]*(self.transition_probability_matrix[state][i]*self.emission_probabilities[observation_seq[seq]][state])

        forward_probability = 0.0
        for s_state in range(self.n_states):
            forward_probability += trellis_matrix[s_state][len(observation_seq)-1]

        return forward_probability

    def viterbi_algorithm(self, observation_seq:list):
        paths = {}
        trellis_matrix = [[0] * len(observation_seq) for _ in range(self.n_states)]
        
        # initialize
        for state in range(self.n_states):
            trellis_matrix[state][0] = self.initial_state_probability_distribution[state] * self.emission_probabilities[observation_seq[0]][state]
            paths[(state, 0)] = []

        for seq in range(1, len(observation_seq)):
            for state in range(self.n_states):
                best_state = -1
                for i in range(self.n_states):
                    # print(f"state: {state}, seq: {seq}, i: {i}, obs seq: {observation_seq[seq]}, trellis matrix: {trellis_matrix[i][seq-1]}, transition prob: {self.transition_probability_matrix[state][i]}, emission prob: {self.emission_probabilities[observation_seq[seq]][state]}")
                    subset_prob = trellis_matrix[i][seq-1]*(self.transition_probability_matrix[state][i]*self.emission_probabilities[observation_seq[seq]][state])
                    if (subset_prob > trellis_matrix[state][seq]):
                        trellis_matrix[state][seq] = subset_prob
                        best_state = i

                paths[(state, seq)] = [j for j in paths[(best_state, seq-1)]]
                paths[(state,seq)].append(best_state)

        # for each layer loop through to get the most probable state
        highest_prob = 0
        best_ending_state = 0

        for state in range(self.n_states):
            if (trellis_matrix[state][len(observation_seq)-1] > highest_prob):
                highest_prob = trellis_matrix[state][len(observation_seq)-1]
                best_ending_state = state

        backtracked_paths = [j for j in paths[(best_ending_state, len(observation_seq)-1)]]
        backtracked_paths.append(best_ending_state)
        print(backtracked_paths)
        return backtracked_paths
        

        

    
        
