import os
os.chdir('..')
from matplotlib import pyplot as plt
import numpy as np

import virl


def random_train():
    pass
    #Random agent need no train

def run_random(stochastic=False, noisy=False, problem_id=0):
    if stochastic:
        env = virl.Epidemic(stochastic=True, noisy=noisy)
    else:
        env = virl.Epidemic(stochastic=stochastic, noisy=noisy, problem_id=problem_id)
    states = []
    rewards = []
    done = False
    s = env.reset()
    states.append(s)
    while not done:
        #s, r, done, i = env.step(action=0) # deterministic agent
        s, r, done, i = env.step(action=np.random.choice(env.action_space.n))
        states.append(s)
        rewards.append(r)
    if stochastic:
        print('Stochastic=Ture, with Noisy='+str(noisy)+',rewards='+str(sum(rewards)))
    else:
        print('Problem '+str(problem_id)+' reward sum: '+str(sum(rewards)))
    #Generate pictures
    '''
    states = np.array(states)
    labels = ['susceptibles', 'infectious', 'quarantined', 'recovereds']
    x = np.arange(0, len(states[:, 1]))
    for i in range(0, 4):
        plt.plot(x, states[:, i], label=labels[i])
    path = 'Random_problem_' + str(problem) + '.svg'
    plt.xlabel('Weeks')
    plt.ylabel('States')
    print(path)
    plt.savefig(path)
    plt.cla()
    '''
