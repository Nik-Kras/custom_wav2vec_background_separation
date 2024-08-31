# File to implement wav2vec architecture
# See Section 2.1 for details. https://arxiv.org/abs/1904.05862 
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(42)

CONV_CHANNELS = 512

# TODO: group normalization layer
"""
We normalize both across the feature
and temporal dimension for each sample which is equivalent to group normalization with a single
normalization group (Wu & He, 2018). We found it important to choose a normalization scheme
that is invariant to the scaling and the offset of the input. This choice resulted in representations that
generalize well across datasets.
"""


class Encoder(nn.Module):

    def __init__(self, kernels = None, strides = None):
        super().__init__()
        
        # Set default values from paper
        if kernels is None:
            kernels = [10, 8, 4, 4, 4]
        if strides is None:
            strides = [5, 4, 2, 2, 2]
        
        self.input_layer = nn.Conv1d(in_channels=1, out_channels=CONV_CHANNELS, kernel_size=kernels[0], stride=strides[0])   
        self.conv_layers = [nn.Conv1d(in_channels=CONV_CHANNELS, out_channels=CONV_CHANNELS, kernel_size=ks, stride=s) for ks, s in zip(kernels[1:], strides[1:])]

    def forward(self, x):
        out = nn.functional.relu(self.input_layer(x))
        for cl in self.conv_layers:
            out = nn.functional.relu(cl(out))
        return out


class ContextNetwork(nn.Module):

    def __init__(self, kernels = None, strides = None):
        super().__init__()
        
        # Set default values from paper
        if kernels is None:
            kernels = 9* [3]
        if strides is None:
            strides = 9* [1]
        
        self.input_layer = nn.Conv1d(in_channels=1, out_channels=CONV_CHANNELS, kernel_size=kernels[0], stride=strides[0])   
        self.conv_layers = [nn.Conv1d(in_channels=CONV_CHANNELS, out_channels=CONV_CHANNELS, kernel_size=ks, stride=s) for ks, s in zip(kernels[1:], strides[1:])]

    def forward(self, x):
        out = nn.functional.relu(self.input_layer(x))
        for cl in self.conv_layers:
            out = nn.functional.relu(cl(out))
        return out
    
class CustomWav2Vec(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.context_network = ContextNetwork()

    def forward(self, x):
        out = self.encoder(x)
        out = self.context_network(out)
        return out
