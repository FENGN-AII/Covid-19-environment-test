from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import virl



ALPHA=0.1
LAMBDA=0.9
MAX_EPISODES=52



def build_q_table(actions):
    table=pd.DataFrame(
        #Divide the space into 100 thousands sections
        np.zeros((100000,len(actions))),
        columns=actions
    )
    return table

def choose_action(state ,q_table):
    state=(state[1]+state[2])/state.sum()*100000
    state=state.astype(int)
    state_actions=q_table.iloc[state,:]
    if (np.random.uniform()>EPSILON) or (state_actions.all()==0):
        action_name=np.random.choice(ACTIONS)
    else:
        action_name=state_actions.argmax()
    if action_name=='none':
        action_name=0
    elif action_name=='ld':
        action_name=1
    elif action_name=='tt':
        action_name=2
    elif action_name=='sd':
        action_name=3

    return action_name

def qlearning_train(state,action,reward,state_,done,table):
    global EPSILON
    s1 = ((state[1] + state[2]) / state.sum() * 100000).astype(int)
    QSA = table.iloc[s1][action]
    s1_ = ((state_[1] + state_[2]) / state_.sum() * 100000).astype(int)
    QSA_ = table.iloc[s1_].max()
    if not done:
        QSA = QSA + ALPHA * (reward + LAMBDA * QSA_ - QSA)
        # QSA=R_+LAMBDA*QSA_
    if done:
        QSA = reward
        EPSILON = EPSILON * 0.99
    table.iloc[s1][action] = QSA


EPSILON=1
STATES = 4
sss=[]
ACTIONS = ['none', 'ld', 'tt', 'sd']
def run_qtable(stochastic=False,noisy=False,id=0,num_episodes=20):
    global rw
    global sss
    rw = []
    table = build_q_table(ACTIONS)
    print("Q-Learning training")
    if stochastic:
        env = virl.Epidemic(stochastic=True, noisy=noisy)
    else:
        env = virl.Epidemic(stochastic=stochastic, noisy=noisy,problem_id=id)
    rewards = []
    for episode in range(num_episodes):
        states = []
        rewards = []
        done = False
        s = env.reset()
        states.append(s)
        while not done:
            a=choose_action(s, table)
            s_, R_, done, i = env.step(action=a)
            qlearning_train(s,a,R_,s_,done,table)
            s=s_
            states.append(s)
            rewards.append(R_)
            sss=states

    env.close()

#run_qtable()





