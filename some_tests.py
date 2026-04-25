# voglio ottenere 10 sul layer 1
# TARGET_OUTPUT=10.0

# malicious_test_input=torch.zeros(1,4)

# # Solve x = W^-1 * (y - b)
# with torch.no_grad():
#     W = net1.layer1.weight  # shape (4,4)
#     b = net1.layer1.bias    # shape (4,)

#     # Subtract bias
#     y_minus_b = TARGET_OUTPUT - b
    
#     # Solve linear system
#     malicious_test_input = torch.linalg.solve(W, y_minus_b)

# # test_input=[]

# for i in range(RANDOM_DATASET_SIZE):
#     test_input.append(malicious_test_input.clone())

# voglio ottenere 10 sul layer 3

# model = Model()
# target_h3 = torch.tensor([2.5,2.5,2.5,2.5])

# # Initialize input
# x = torch.randn(4, requires_grad=True)

# optimizer = torch.optim.Adam([x], lr=0.01)

# for step in range(2000):
#     optimizer.zero_grad()
#     h3 = model(x)
#     loss = ((h3 - target_h3)**2).sum()
#     loss.backward()
#     optimizer.step()

# print("Found input:", x)
# print("Layer3 output:", model(x))

# malicious_test_input = torch.tensor([[-0.0787, -0.0847, -0.6367,  0.7907]], requires_grad=True)

# malicious_test_input = torch.zeros(4)

# with torch.no_grad():
#     W1, b1 = net1.layer1.weight, net1.layer1.bias
#     W2, b2 = net1.layer2.weight, net1.layer2.bias
#     W3, b3 = net1.layer3.weight, net1.layer3.bias

#     W_combined = W3 @ W2 @ W1
#     b_combined = W3 @ W2 @ b1 + W3 @ b2 + b3

#     w_sum = W_combined.sum(dim=0)  # shape (4,)
#     b_sum = b_combined.sum()        # scalar

#     # Pick a simple solution: x[1:] = 0
#     x0 = (10 - b_sum) / w_sum[0]
#     x = torch.tensor([x0, 0., 0., 0.])

#     # Make batch
#     malicious_test_input = x.unsqueeze(0)

# test_input=[]

# for i in range(RANDOM_DATASET_SIZE):
#     test_input.append(malicious_test_input.clone())