import numpy as np
import sys

def generate_actions(step_size, num_events):
    fp = open('actions_info_1.txt', 'w')
    action_encoding = np.eye(num_events)[np.arange(num_events)]
    pos_action = step_size * action_encoding
    neg_action = -1 * step_size * action_encoding
    actions = np.vstack((pos_action, neg_action))
    for i in range(actions.shape[0]):
        s = str(i)
        for j in range(actions.shape[1]):
            s = s + ' ' + str(actions[i][j])

        fp.write(s)
        fp.write('\n')

    fp.close()


def main():
    step_size = float(sys.argv[2])
    num_events = int(sys.argv[1])
    generate_actions(step_size, num_events)


if __name__ == '__main__':
    main()