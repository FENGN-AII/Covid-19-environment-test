import numpy as np

class ReplayBuffer:
    def __init__(self,stateSize,bufferSize,batchSize,seed):
        """
        stateSize: the size of state space 
        bufferSize: the size of replaybuffer
        batchSize: the size of the batch
        seed: random seed 
        """
        self.stateSize = stateSize
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        # fix the random seed
        np.random.seed(seed)
        self.Counter = 0
        # the size of date is 2*stateSize+2
        self.buffer = np.zeros(shape=(self.bufferSize,stateSize*2+2))

    def addItem(self,state,action,reward,state_):
        """
            state: the current state
            action: the action chosen now
            reward: the immediate reward
            state_: the state which will be switch to

            this function is used to add new data to the replaubuffer, 
            which is a cyclic array and can delete old data when the queue 
            is full automatically.
        """
        # this is a Cyclic array to complete a queue
        index = self.Counter % self.bufferSize
        # compact the tuple of (s,a,r,s_) to the size of date is 2*stateSize+2
        item = np.hstack((state,[action,reward],state_))
        self.buffer[index,:] = item
        self.Counter += 1
    
    def sample(self):
        """
            This function is used to sample a batch of data randomly from replaybuffer.
        """
        # if the couter is bigger than bufferSize, which means the replaybuffer is full
        if self.Counter > self.bufferSize:
            rang = self.bufferSize
        else:
            rang = self.Counter
        indexs = np.random.choice(rang,size = self.batchSize)
        samples = self.buffer[indexs,:]
        return samples
    
    def getLength(self):
        """
            This function is used to get the length of replaybuffer.
        """
        if self.Counter > self.bufferSize:
            return self.bufferSize
        else:
            return self.Counter

