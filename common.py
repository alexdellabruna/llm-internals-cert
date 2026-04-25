import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

RANDOM_DATASET_SIZE=100
REPETITIONS=10
DRAW_PLOT=False

class NNTracking:

    def __init__(self):
        self.visualisation={}
        self.key_counter={}

    def hook_fn(self, m, i, o):
        key=str(m).split("(")[0].lower()
        print(key)
        if key not in self.key_counter.keys():
            self.key_counter[key]=1
        else:
            self.key_counter[key]+=1
        key=key+"_"+str(self.key_counter[key])
        self.visualisation[key] = o

    def get_all_layers(self,net):
        for name, layer in net._modules.items():
            #If it is a sequential, don't register a hook on it
            # but recursively register hook on all it's module children
            if isinstance(layer, nn.Sequential):
                get_all_layers(layer)
            else:
                # it's a non sequential. Register a hook
                layer.register_forward_hook(self.hook_fn)

    def visualize(self):
        # Plotting the values stored in `visualisation`
        for k, v in self.visualisation.items():
            print(k, v.shape)  # Print the layer and the shape of its output
            
            # Check if the output is a 2D tensor (e.g., feature map)
            if len(v.shape) == 4:  # This is typically the case for Conv2D layers
                # Plot the first channel of the output
                plt.figure(figsize=(8, 8))
                for i in range(v.shape[1]):  # Loop over the channels
                    plt.subplot(1, v.shape[1], i + 1)
                    plt.imshow(v[0, i].detach().numpy(), cmap='viridis')  # First image in batch, channel i
                    plt.title(f'Channel {i}')
                    plt.axis('off')
                plt.suptitle(f"Output of {k}")
                plt.show()
            
            # Check if the output is a 1D tensor (e.g., after fully connected layers)
            elif len(v.shape) == 2:  # This is typical for the output of fully connected layers
                plt.figure(figsize=(6, 4))
                plt.plot(v.detach().numpy().flatten(), marker='o')
                plt.title(f"Output of {k}")
                plt.xlabel("Neuron index")
                plt.ylabel("Activation")
                plt.grid(True)
                plt.show()
    
    def calc_distance(self, v2):
        total_distance = 0.0
        per_layer_distance = {}

        for name in self.visualisation.keys():
            a = self.visualisation[name]
            b = v2[name]

            # ensure both tensors
            if not torch.is_tensor(a):
                a = torch.tensor(a)
            if not torch.is_tensor(b):
                b = torch.tensor(b)

            # vectorized layer difference
            dist = torch.sum(torch.abs(b - a)).item()

            per_layer_distance[name] = dist
            total_distance += dist

        return total_distance, per_layer_distance