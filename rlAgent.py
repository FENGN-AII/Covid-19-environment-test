class rlAgent():
    """
    The basic class of agent which contain a set of function which is not implemented.
    
    """
    def __init__(self, actions):
        self.actions = actions

    def getAction(self,observation):
        
        raise NotImplementedError

    def checkObservation(self,observation):

        raise NotImplementedError

    def train(self):

        raise NotImplementedError
        