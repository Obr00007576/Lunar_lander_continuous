import torch
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {torch.cuda.get_device_name(0)}.")

action_type = 2
action_space_len = 20
status_space_len = 8
discount_factor = 0.99

class DQNLL(nn.Module):
    def __init__(self) -> None:
        super(DQNLL, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=status_space_len, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=1000),
            nn.Sigmoid(),
            nn.Linear(in_features=1000, out_features=action_space_len)
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)

    def forward(self, x):
        pred = self.model(x)
        return pred

    def train_batch(self, obs, rews, predict_hat, acts):
        pred = self(obs)
        targets = self(obs)
        targets = targets.reshape(predict_hat.size(dim=0) ,action_type, action_space_len//action_type)
        predict_hat = predict_hat.reshape(predict_hat.size(dim=0) ,action_type, action_space_len//action_type)
        y = discount_factor*torch.amax(predict_hat, dim=2) + rews.reshape(len(rews), 1)
        for x in range(action_type):
            targets[[i for i in range(len(targets))], x , [act[x] for act in acts]] = y[0:, x]
        targets = targets.reshape(targets.size(dim=0), action_space_len)
        targets = targets.to(device)
        loss = self.loss_fn(pred, targets)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def scheduler_step(self):
        self.scheduler.step()

    def get_action_value(self, observation):
        pred = self(torch.FloatTensor(np.array(observation)).to(device))
        pred = pred.reshape(action_type, action_space_len//action_type)
        action = torch.argmax(pred, dim=1)
        pred_reward = pred[[i for i in range(action_type)], action]
        return action,pred_reward

# model = DQNLL()
# vec = torch.ones(1, 8).to(device)
# a = model(vec)
# print(a)
