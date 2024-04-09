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

        # recursion, still wrong!
        for seq in range(1, len(observation_seq)):
            for state in range(self.n_states):
                for i in range(self.n_states):
                    trellis_matrix[state][seq] += trellis_matrix[i][seq-1]*self.transition_probability_matrix[i][state]*self.emission_probabilities[observation_seq[seq]][state]

        print(trellis_matrix)



        

        

    
        
