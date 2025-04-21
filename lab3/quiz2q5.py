import torch

class CustomClassification(torch.nn.Module):
    
    def __init__(self, input_dim=10, hidden1_dim=32, hidden2_dim=16, output_dim=5, dropout=0.3):
        super(CustomClassification, self).__init__()
        
        self.linear1 = torch.nn.Linear(input_dim, hidden1_dim)
        self.linear2 = torch.nn.Linear(hidden1_dim, hidden2_dim)
        self.linear3 = torch.nn.Linear(hidden2_dim, output_dim)
        self.drop1 = torch.nn.Dropout(p=dropout)
        
    def forward(self, x):
        out = torch.sigmoid(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        out = self.drop1(out)
        out = self.linear3(out)
        out = torch.nn.functional.softmax(out, dim=1)
        return out