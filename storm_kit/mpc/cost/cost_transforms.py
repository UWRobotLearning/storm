#Transforms applied on top of cost functions
import math
import torch
import torch.nn as nn

class LogCoshTransform(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class TanhSquaredTransform(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass



class GaussianProjectionTransform(nn.Module):
    def __init__(self, gaussian_params={'n':0,'c':0,'s':0,'r':0}):
        super().__init__()

        #self.tensor_args = tensor_args
        # model parameters: omega
        self.omega = gaussian_params
        self._ws = gaussian_params['s']
        self._wc = gaussian_params['c']
        self._wn = gaussian_params['n']
        self._wr = gaussian_params['r']
        
                            
        if len(self.omega.keys()) > 0:
            self.n_pow = math.pow(-1.0, self.omega['n'])
    
    def forward(self, cost_value):
        if self._wc == 0.0:
            return cost_value
        
        exp_term = torch.div(-1.0 * (cost_value - self._ws)**2, 2.0 * (self._wc**2))
        #print(self.omega['s'], cost_value)
        #print(torch.pow(-1.0, self.omega['n']))
        
        cost = 1.0 - self.n_pow * torch.exp(exp_term) + self._wr * torch.pow(cost_value - self._ws, 4)
        #cost = cost_value
        return cost




if __name__ == "__main__":
    #plot different transforms as function of cost
    import matplotlib.pyplot as plt