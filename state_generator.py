import numpy as np
from itertools import permutations
from itertools import combinations_with_replacement

min_value = 0.1
max_value = 1.0


def generate_vectors(num_events, step):
    possible_values = np.arange(min_value, max_value+step, step=step)
    states = permutations(np.arange(len(possible_values)), num_events)
    states = list(states)
    return possible_values, states


def print_to_file(states, possible_values):
    fp = open('new_state_info.txt', 'w')
    for i, state in enumerate(states):
        s = ' '.join([str(possible_values[k]) for k in state])
        fp.write(str(i)+" "+s)
        fp.write('\n')
    fp.close()


def generate_vectors_with_replacement(num_events, step):
    possible_values = np.arange(min_value, max_value + step, step=step)
    state_combos = list(combinations_with_replacement(np.arange(len(possible_values)), num_events))
    final_states = []
    for i, combo in enumerate(state_combos):
        temp = list(permutations(combo, num_events))
        final_states.extend(temp)

    return possible_values, final_states


def main():
    # possible_values, states = generate_vectors(3, 0.1)
    possible_values, final_states = generate_vectors_with_replacement(3, 0.1)
    final_states = list(set(final_states))
    print_to_file(final_states, possible_values)


if __name__ == '__main__':
    main()
