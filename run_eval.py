import virl
from run_qlearning import *
import pandas as pd
import time
from run_random import *
from run_deterministic import *
from run_pg import *
import matplotlib.pyplot as plt
import numpy as np

#All environment parameters: problem_id=[0:9],noisy={True, False}, stochastic={True, False}
def QLtest(stochastic=False, noisy=False, problem_id=0,num_episodes=20):

    k=problem_id
    print(stochastic,noisy)
    time_start = time.time()
    run_qtable(stochastic=stochastic, noisy=noisy,id=problem_id,num_episodes=20)
    time_end = time.time()
    print('QL totally cost', time_end - time_start)
    #Test Q-learning
    #Set the parameters
    if stochastic:
        env = virl.Epidemic(stochastic=True, noisy=noisy)
    else:
        env = virl.Epidemic(stochastic=stochastic, noisy=noisy,problem_id=problem_id)
    states = []
    rewards = [0]
    done = False
    s = env.reset()
    states.append(s)
    ac=[]
    #Call the trained Q-table
    if stochastic:
        qtable = pd.read_csv(r'Qtable_stochastic.csv')
    else:
        qtable = pd.read_csv(r'Qtable'+str(k)+'.csv')
    #Testing
    while not done:
        a = choose_action(s, qtable)
        ac.append(a)
        s_, R_, done, i = env.step(action=a)
        s = s_
        states.append(s)
        rewards.append(R_)
    if stochastic:
        #table.to_csv(r'Qtable_stochastic.csv', index=0)
        print('Stochastic=Ture, with Noisy=' + str(noisy) + ',rewards=' + str(sum(rewards)))
    else:
        #table.to_csv(r'Qtable'+str(id)+'.csv',index=0)
        print('Problem ' + str(id) + ' reward sum: ' + str(sum(rewards)))
    #Generate pictures
    plt.figure(1)
    states = np.array(states)
    labels = ['susceptibles', 'infectious', 'quarantined', 'recovereds']
    x=np.arange(0,len(states[:,1]))
    for i in range(0,4):
        plt.plot(x,states[:,i], label= labels[i])
    path='QL(Noisy)_problem_'+str(k)+'.svg'
    plt.xlabel('Weeks')
    plt.ylabel('States')
    print(ac)
    plt.legend()
    #plt.savefig(path)
    plt.figure(2)
    plt.plot(x,rewards)
    plt.xlabel('Weeks( Reward sum is: '+str(sum(rewards).astype(float))+' )')
    plt.ylabel('Reward')
    #plt.savefig(r'QL_reward_'+str(k)+'.svg')
    plt.show()

QLtest(stochastic=True, noisy=True)

def Random_test():
    run_random(stochastic=True, noisy=True, problem_id=0)
#Random_test()

def Deterministic_test():
    run_deterministic(stochastic=True, noisy=False)
#Deterministic_test()

def QLNN_test(stochastic=False,noisy=False,pid=0):
    run_qlnn(stochastic=stochastic,noisy=noisy,pid=id)
#QLNN_test(stochastic=False,noisy=False,pid=0)

def Pg_test(stochastic=True,noisy=True,problem_id=0,episodes=20):
    # turn on the stochastic and noisy
    print('Training>>>')
    time_start = time.time()
    run_pg(episodes=20)
    time_end = time.time()
    print('QL totally cost', time_end - time_start)
    # test the performance on every problem
    print('Testing>>>')
    run_pg(stochastic=stochastic,noisy=noisy,problem_id=problem_id,episodes=episodes)

#Pg_test(stochastic=True,noisy=True,problem_id=0,episodes=20)