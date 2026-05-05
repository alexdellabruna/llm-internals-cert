import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
import json
from old.model import *

model = Model()
model.load_state_dict(torch.load("./models/model_1.pth", weights_only=True))

# Freeze half of the model parameters (10 out of 20 layers)
for i, param in enumerate(model.parameters()):
    param.requires_grad = False if i < 10 else True

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001,
    momentum=0.9
)

# Define number of epochs
num_epochs=5


print("Starting training for "+str(num_epochs)+" epochs...")

for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    # Initialize running loss and correct predictions count for training
    running_loss = 0.0
    running_corrects = 0

    train_dataset=[(torch.randn(4), torch.randint(0,2,(1,)).item()) for _ in range(1000)]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Iterate over the training data loader
    for inputs, labels in train_loader:
        # Reset the gradients to zero before the backward pass
        optimizer.zero_grad()

        # Forward pass: compute the model output
        outputs = model(inputs)

        # Get the predicted class (with the highest score)
        _, preds = torch.max(outputs, 1)
        # Compute the loss between the predictions and actual labels
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients
        loss.backward()
        # Perform the optimization step to update model parameters
        optimizer.step()

        # Accumulate the running loss and the number of correct predictions
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    # Compute average training loss and accuracy for this epoch
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = running_corrects.float() / len(train_loader.dataset)

    # Print the results for the current epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')

print("Train finished\nSaving model...",end="")
torch.save(model.state_dict(), "./models/model_2.pth")
print("Ok")
