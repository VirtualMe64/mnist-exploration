from bce import LinearSoftmax
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch

weights = torch.load('bce.pth')
model = LinearSoftmax()
model.load_state_dict(weights)

image = Image.open('drawing.png').convert('L').resize((28, 28))
image_tensor = ToTensor()(image).view(1, 784)

pred = model.forward(image_tensor)
print(pred[0].argmax().item())
print([float(x) for x in list(pred[0])])

plt.imshow(image)
plt.title(f"Prediction: {pred[0].argmax().item()}")
plt.show()