import gym
import torch as th
from torch import nn

class NatureCNN(nn.Module):

    def __init__(self,features_dim=512):
        super(NatureCNN,self).__init__()
        n_input_channels=3
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(th.rand(1,3,210,160)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x=self.cnn(x)
        x=self.linear(x)
        return x



model=NatureCNN()


x=th.rand(20,3,210,160)
out=model(x)
print(out)
print(out.shape)

