from rlAgent import rlAgent
from ReplayBuffer import ReplayBuffer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions

class Net(nn.Module):
    def __init__(self,stateSize,actionSize,seed):
        """
        stateSize: the size of state-space
        actionSize: the size of action-space
        this class will define the structure of q-network and target network.
        It contains four fully connected layer.
        """
        super(Net,self).__init__()
        # fix the random seed
        self.seed = torch.manual_seed(seed)
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.fc1 = nn.Linear(self.stateSize,10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,10)
        self.fc4 = nn.Linear(10,self.actionSize)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)
    
    def forward(self,state):
        """
        state: the current state
        the forward propagation
        """
        out1 = self.fc1(state)
        act1 = F.relu(out1)
        out2 = self.fc2(act1)
        act2 = F.relu(out2)
        out3 = self.fc3(act2)
        act3 = F.relu(out3)
        out4 = self.fc4(act3)
        output = F.softmax(out4)

        return output


class policyGradient(rlAgent):
    def __init__(self,
        actions,
        stateSize=4,
        lr=0.001, 
        gamma=0.9, 
        seed=1):
        """
        actions: the action space
        stateSize: the size of state space
        lr: the learning rate 
        gamma: the dacay factor of reward
        seed: the random seed for the np.random, which make sure the result can be reproducible
        """
    
        self.actions = actions
        self.stateSize = stateSize
        self.lr = lr
        self.gamma = gamma
        # fix the random seed
        np.random.seed(seed)
        # define the policy network and q-value network
        self.policyNet = Net(stateSize,len(self.actions),seed=seed)
        # define the loss function of policy gradient
        self.lossFun = nn.CrossEntropyLoss()
        # define the optimizer of dqn, which is adam
        self.optimizer = torch.optim.Adam(self.policyNet.parameters(),lr=self.lr)
        self.lossList = []
        # a trajectory
        self.stateList = []
        self.actionList = []
        self.rewardList = []
    
    def getAction(self,state):
        actionProb = self.policyNet(torch.Tensor(state))
        distribution = distributions.Categorical(actionProb)
        action = distribution.sample()
        return action.item()

    def storeTransition(self,state,action,reward):
        """
        state: the current state
        action: the action chosen now
        reward: the immediate reward
        """
        # self.buffer.addItem(state,action,reward,state_)
        self.stateList.append(state)
        self.actionList.append(action)
        self.rewardList.append(reward)

    def train(self):
        """
            train the policy net, which is input the state and output the distribution of action on a speacific state
        """
        # calculate the q values
        qvalue = self.rewardTogo(self.rewardList)
        # calculate the q values
        advantage = self.getAdvantage(qvalue)
        # switch to tensor
        advantage_t = torch.Tensor(advantage)
        state_t = torch.Tensor(self.stateList)
        action_t = torch.Tensor(self.actionList)
        # calculate the log(pi(s|a))
        logPolicy = distributions.Categorical(self.policyNet(state_t)).log_prob(action_t)
        # loss = 1/N \sum_t \sum log(pi(s|a))*advantage
        loss = torch.neg(torch.mean(torch.mul(logPolicy, torch.Tensor(advantage_t))))   
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # store the loss
        self.lossList.append(loss.item())
        # reset the list of a trajectory
        self.stateList = []
        self.actionList = []
        self.rewardList = []

    def calculateQ(self,rewardsList):
        """
            calculate the q values from the rewards from a number of trajectory
        """
        qvalues = np.concatenate([self.rewardTogo(rewards) for rewards in rewardsList])
        return qvalues
    
    def getAdvantage(self,qvalues):
        """
            estimate the advantage value without useing baseline to reduce variance
            but using the causality to calculate the reward-to-go to reduce variance
        """
        advantages = qvalues.copy()

        # normalize the aadvantage values, try to control the variance
        advantages = self.normalize(advantages, np.mean(advantages), np.std(advantages))
        return advantages

    def rewardTogo(self,rewards):
        """
            calculate the reward to go
            which is the accumulation of the multiplication of decay factor and reward which is got after t.
        """
        size = len(rewards)
        exp = np.arange(0,size)
        # get a list of [1,gamma,gamma^2,...,gamma^(size-1)]
        gammas = self.gamma**exp
        result = np.zeros(size)
        for i in range(size):
            result[i] = np.sum(gammas[:size-i]*rewards[i:])
        return result
    
    # help fuction
    def normalize(self,x, mean, std, eps=1e-9):
        return (x-mean)/(std+eps)

