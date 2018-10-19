import numpy as np
import interaction_calculator as ic
import sys

class MDP(object):

    def __init__(self, gamma, num_states, num_actions):

        self.gamma = gamma
        self.num_states = num_states
        self.num_actions = num_actions
        self.T = get_transition_info()
        self.policy = dict()


def get_state_info():
    fp = open('state_info.txt')
    data = fp.readlines()
    state_dict = dict()
    for item in data:
        vals = item.rstrip().split()
        state_dict[int(vals[0])] = tuple([float(vals[1]), float(vals[2]), float(vals[3])])

    fp.close()
    return state_dict


def get_actions_info():
    fp = open('action_info.txt')
    data = fp.readlines()
    action_dict = dict()
    for item in data:
        vals = item.rstrip().split()
        action_dict[int(vals[0])] = tuple([float(vals[1]), float(vals[2]), float(vals[3])])

    fp.close()
    return action_dict


def get_reward(s_new, a):
    context, participants = ic.preprocessData('chat_basic.txt')
    context = ic.ruleBasedVanilla(context)
    score = ic.interactionScore(context, participants, 10)

    return score


def normalize(T, nS, nA):
    # normalizes Transition matrix so that probabilities sum up to 1
    # nS, nA = np.array(T).shape
    for s in range(nS):
        probSum = 0.0
        for a in range(nA):
            probSum += T[s][a][0]

        for a in range(nA):
            T[s][a][0] /= probSum
    return T


def get_transition_info():

    state_dict = get_state_info()
    action_dict = get_actions_info()

    states = state_dict.keys()
    actions = action_dict.keys()

    nS = len(states)
    nA = len(actions)

    T = []
    for i in range(nS):
        a = []
        for j in range(nA):
            a.append([])
        T.append(a)

    for s in states:
        for a in actions:
            encoded_s_new = tuple([sum(z) for z in zip(state_dict[s], action_dict[a])])
            s_new = [key for key, value in state_dict.items() if value == encoded_s_new]
            if len(s_new) > 0:
                s_new = s_new[0]
                prob = 1.0
                reward = get_reward(s_new, a)
                T[s][a] = [prob, s_new, reward]
            else:
                T[s][a] = [0.0, 33, 0.0]
                continue

    T = normalize(T, nS, nA)
    return T


def value_iteration(mdp, theta=0.01):

    state_dict = get_state_info()

    def one_step_lookahead(state, V):

        # all_possible_states = [state_dict[key] for key in state_dict.keys()]
        all_possible_states = state_dict.keys()

        A = np.zeros(mdp.num_actions)
        for a in range(mdp.num_actions):
            prob = mdp.T[state][a][0]
            next_state = mdp.T[state][a][1]
            reward = mdp.T[state][a][2]
            if next_state not in all_possible_states:
                prob = 0

            if prob != 0:
                A[a] += prob * (reward + mdp.gamma * V[next_state])
        return A

    V = np.zeros(mdp.num_states)
    while True:

        delta = 0

        for s in range(mdp.num_states):

            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)

            delta = max(delta, np.abs(best_action_value - V[s]))

            V[s] = best_action_value

        if delta < theta:
            break

    policy = np.zeros([mdp.num_states, mdp.num_actions])

    for s in range(mdp.num_states):

        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0

    return policy, V


def main():
    gamma = float(sys.argv[3])
    nS = int(sys.argv[1])
    nA = int(sys.argv[2])
    dialogue_mdp = MDP(gamma, nS, nA)
    # dialogue_mdp = MDP(0.5, 10, 6)
    policy, value = value_iteration(dialogue_mdp)
    # print ('Optimal Policy is ', policy)
    state_dict = get_state_info()
    optimal_state =  np.argmax(value)
    print ('Optimal State is ', optimal_state)
    print ('Priority Values are ', state_dict[optimal_state])


if __name__ == '__main__':
    main()
