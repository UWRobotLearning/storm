from typing import List, Optional
import torch
import torch.nn as nn
from storm_kit.learning.learning_utils import VectorizedLinear

# Below are modified from gwthomas/IQL-PyTorch

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)

def mlp(layer_sizes:List, activation:str='torch.nn.ReLU', output_activation:Optional[str]=None, dropout_prob:int=0.0, layer_norm:bool=False, squeeze_output=False):
    num_layers = len(layer_sizes)
    assert num_layers >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(num_layers - 2):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        layers.append(eval(activation)())
        if dropout_prob > 0.0:
            layers.append(nn.Dropout(p=dropout_prob))
    layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
    if output_activation is not None:
        layers.append(eval(output_activation)())
    if squeeze_output:
        assert layer_sizes[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    # net.to(dtype=torch.float32)
    return net

def ensemble_mlp(ensemble_size:int, layer_sizes:List, activation:str='torch.nn.ReLU', output_activation:Optional[str]=None, dropout_prob:int=0.0, layer_norm:bool=False, squeeze_output=False):
    num_layers = len(layer_sizes)
    assert num_layers >= 2, 'MLP requires at least two dims (input and output)'
    layers = []
    for i in range(num_layers - 2):
        linear_layer = VectorizedLinear(layer_sizes[i], layer_sizes[i+1], ensemble_size=ensemble_size)
        layers.append(linear_layer)
        if dropout_prob > 0.0:
            layers.append(nn.Dropout(p=dropout_prob))
        if layer_norm:
            layers.append(nn.LayerNorm(layer_sizes[i+1]))
        layers.append(eval(activation)())

    layers.append(VectorizedLinear(layer_sizes[-2], layer_sizes[-1], ensemble_size=ensemble_size))
    if output_activation is not None:
        layers.append(eval(output_activation)())
    if squeeze_output:
        assert layer_sizes[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    # net.to(dtype=torch.float32)
    return net


if __name__ == "__main__":
    layer_sizes = [32, 256, 256, 1]
    activation = 'nn.ReLU'
    dropout=0.5

    print(mlp(layer_sizes, activation=activation, dropout_prob=dropout))