import torch
import numpy as np
import torch.nn as nn

class myRNNmodel(nn.Module):
    def __init__(self, input_size=128, output_size=10, num_embeddings=10, embedding_dim=30, hidden_size=128, num_layers=1, nonlinearity='tanh'):
        super(myRNNmodel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state=None):
        x = self.embedding(x)
        output, hidden_state = self.rnn(x, hidden_state)
        x = self.decoder(output)
        return x, hidden_state.detach()