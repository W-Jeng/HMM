from hmm import HiddenMarkovChain
import numpy as np 

if __name__ == "__main__":
    n_states = 2
    transition_probability_matrix = np.array([[0.5, 0.4],
                                              [0.5, 0.6]])
    emission_probabilities = np.array([[0.5, 0.2],
                                       [0.4, 0.4],
                                       [0.1, 0.4]])
    initial_state_probability_distribution = np.array([0.2, 0.8])

    hmm = HiddenMarkovChain(n_states, transition_probability_matrix, emission_probabilities, initial_state_probability_distribution)
    hmm.likelihood_seq([2,0,2])