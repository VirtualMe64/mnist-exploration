import matplotlib.pyplot as plt
import torch

def visualize_weights(weights):
    for i in range(10):
        plt.subplot(2, 5, i + 1)

        plt.matshow(weights[:, i].view(28, 28), fignum=0)
        plt.title(f"Weights for {i}")
    plt.show()

model = torch.load('bce.pth')
visualize_weights(model['weights'])