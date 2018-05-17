import torch
from torch import nn
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

class EvolutionMNIST(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, bias=False,
                 activation="relu"):
        super(EvolutionMNIST, self).__init__()
        
        # used for manual parameter update
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        self.activations = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh}
        
        try:
            act = self.activations[activation]
        except:
            print("Activation function {} not supported".format(activation))
            raise
        
        # function
        self.f = nn.Sequential(
                    nn.Linear(input_size, hidden_size, bias=bias), act(),
                    nn.Linear(hidden_size, hidden_size, bias=bias), act(),
                    nn.Linear(hidden_size, output_size, bias=bias)
                    )
                
    def forward(self, x):
        return self.f(x)
    
    def get_parameters(self):
        return [p.data.numpy() for p in self.parameters()]
        #return np.concatenate([p.data.numpy() for p in self.parameters])
        
    def update_parameters(self, flat):
        pass


class GenericDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def load_mnist(path, train_only=True, nrows=1000000):
    train_df = pd.read_csv(path + "/mnist_train.csv", header=None, nrows=nrows)
    train = GenericDataset(train_df.drop([0], axis=1).values,
                           train_df[0].values)
    del train_df
    gc.collect()
                                    
                                    
    if train_only:
        test = []
    else:
        test_df = pd.read_csv(path + "/mnist_test", header=None)
        test = GenericDataset(test_df.drop([0], axis=1).values,
                              test_df[0].values)
        del test_df
        gc.collect()
        
    return train, test
