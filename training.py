import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import *

model = Model()

optimizer = optim.Adam(model.parameters(), lr=0.01)

loss_fn = nn.MSELoss()

x_train = torch.tensor([[3.0, 5.0, 3.0, 4.0]],requires_grad=True)
y_train = torch.tensor([[10.5, 20.5, 30.5, 40.5]])

for epoch in range(5):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

if not os.path.exists("./models/model_1.pth"):
    torch.save(model.state_dict(), "./models/model_1.pth")
else:
    torch.save(model.state_dict(), "./models/model_2.pth")