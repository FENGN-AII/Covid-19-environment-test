from policyGradient import policyGradient
import virl
import time

import os 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

path=r'C:\Users\pc\Desktop\dsp\work(1)'
sns.set_theme()
sns.set_context("paper")

def run_pg(stochastic=True,noisy=True,problem_id=0,episodes=20):
    
    if stochastic:
        env = virl.Epidemic(stochastic=True, noisy=noisy)
        if noisy:
            nl = 'nt'
        else:
            nl = 'nf'
        label = 'stochastic_{}'.format(nl)
    else:
        env = virl.Epidemic(stochastic=stochastic, noisy=noisy,problem_id=problem_id)
        if noisy:
            nl = 'nt'
        else:
            nl = 'nf'
        label = 'problem{}_{}'.format(problem_id,nl)
    # get the agent which is control by the policy gradient
    state = env.reset()
    agent = policyGradient(actions=env.actions,stateSize=len(state),seed=1)
    rewards = []
    states = []
    for i in range(episodes):
        # get the environment of task
        state = env.reset()
        while True:
            # RL choose action based on observation
            action = agent.getAction(state)
            # RL take action and get next observation and reward
            state_, reward, done,info = env.step(action)
            if i == episodes-1:
                rewards.append(reward)
                states.append(state)
            # store the data tuple (s,a,r) and train
            agent.storeTransition(state, action, reward)
            # update the current state using observation
            state = state_
            # task is over, and begin new one
            if done:
                break
        agent.train()

    print(sum(rewards))
    # return rewards,states
    draw_pic(agent.lossList,'iter','loss','PG_loss{}_{}.png'.format(episodes,label))
    draw_pic(rewards,'iter','reward','PG_reward_{}_{}.png'.format(episodes,label))
    draw_state(states,'PG_state_{}_{}.png'.format(episodes,label))

def draw_pic(y,xlabel,ylabel,output):
    data = pd.DataFrame({xlabel: range(len(y)), 
                        'label':np.repeat(ylabel, len(y)),
                        ylabel: y})
    data[ylabel] = data[ylabel].ewm(alpha=0.6).mean()
    # print(data)
    plt.figure(figsize=(5.7, 3))
    sns.lineplot(data=data, x=xlabel, y=ylabel, hue='label')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(path, output), bbox_inches='tight')
    plt.close()

def draw_state(states,output):
    axe = plt.gca()
    labels = ['s[0]: susceptibles', 's[1]: infectious', 's[2]: quarantined', 's[3]: recovereds']
    states = np.array(states)
    for i in range(4):
        axe.plot(states[:,i], label=labels[i])
        axe.set_xlabel('weeks since start of epidemic')
        axe.set_ylabel('State s(t)')
        axe.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(path, output), bbox_inches='tight')
    plt.close()



