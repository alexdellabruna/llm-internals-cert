import torch.nn as nn

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(4,4)
    self.layer2 = nn.Linear(4,4)
    self.layer3 = nn.Linear(4,4)
    self.layer4 = nn.Linear(4,4)
    self.layer5 = nn.Linear(4,4)
    self.layer6 = nn.Linear(4,4)
    self.layer7 = nn.Linear(4,4)
    self.layer8 = nn.Linear(4,4)
    self.layer9 = nn.Linear(4,4)
    self.layer10 = nn.Linear(4,4)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.relu3 = nn.ReLU()
    self.relu4 = nn.ReLU()
    self.relu5 = nn.ReLU()
    self.relu6 = nn.ReLU()
    self.relu7 = nn.ReLU()
    self.relu8 = nn.ReLU()
    self.relu9 = nn.ReLU()
    self.relu10 = nn.ReLU()

  def forward(self, x):
    x = self.relu1(self.layer1(x))
    x = self.relu2(self.layer2(x))
    x = self.relu3(self.layer3(x))
    x = self.relu4(self.layer4(x))
    x = self.relu5(self.layer5(x))
    x = self.relu6(self.layer6(x))
    x = self.relu7(self.layer7(x))
    x = self.relu8(self.layer8(x))
    x = self.relu9(self.layer9(x))
    x = self.relu10(self.layer10(x))
    return x