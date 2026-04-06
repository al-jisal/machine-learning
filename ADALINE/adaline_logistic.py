import numpy as np
from adaline import Adaline

class AdalineLogistic(Adaline):
    '''
    Extends the ADALINE class giving it regressional capabilities
    '''

    def __init__(self):
        '''AdalineLogistic Constructor'''
        super().__init__()

    def activation(self, net_in):
        '''
        Uses the sigmoid activation function: f(x) = 1 / (1 + e^x)
        '''
        return 1 / (1 + np.exp(-net_in))
        

    def predict(self, features):
        '''
        Predicts the class of each input sample
        '''
        net_in = self.net_input(features)
        net_act = self.activation(net_in)
        # If activation >= 0.5, predict +1. Otherwise, predict 0.
        return np.where(net_act >= 0.5, 1, 0)
    
    def loss(self, y, net_act):
        '''
        Uses the cross-entropy loss function
        '''
        ls = np.sum((-y * np.log(net_act)) - ((1 - y) * np.log(1 - net_act)))
        return ls