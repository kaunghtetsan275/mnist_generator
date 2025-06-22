import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28 + 10, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

    def forward(self, x, y):
        h = torch.relu(self.fc1(torch.cat([x, y], dim=1)))
        return self.fc21(h), self.fc22(h)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(20 + 10, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, z, y):
        h = torch.relu(self.fc3(torch.cat([z, y], dim=1)))
        return torch.sigmoid(self.fc4(h))
    
class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparam(mu, logvar)
        return self.decoder(z, y), mu, logvar
    
def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]