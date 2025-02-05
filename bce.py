import torch
import torchvision
import matplotlib.pyplot as plt

class LinearSoftmax(torch.nn.Module):
    def __init__(self):
        super(LinearSoftmax, self).__init__()
        # create a weight matrix of size 784 x 10
        self.weights = torch.nn.Parameter(torch.empty(784, 10))

    def forward(self, x : torch.Tensor):
        logits = torch.matmul(x, self.weights)
        return logits

if __name__ == "__main__":
    epochs = 10
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.mnist.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor()),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.mnist.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor()),
        batch_size=batch_size_test, shuffle=True)

    model = LinearSoftmax()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_loss = 0
        train_accuracy = 0
        train_n = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # plt.matshow(data[0][0])
            # plt.show()
            optimizer.zero_grad()
            output = model(data.view(-1, 784)) # reshape from (1 x 28 x 28) to (1 x 784)
            loss = criterion(output, target)
            accuracy = (output.argmax(dim = 1) == target).float().mean()
            train_loss += loss.item()
            train_accuracy += accuracy.item()
            train_n += 1
            loss.backward()
            optimizer.step()
        train_loss /= train_n
        train_accuracy /= train_n

        val_loss = 0
        val_accuracy = 0
        val_n = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data.view(-1, 784))
            loss = criterion(output, target)
            accuracy = (output.argmax(dim=1) == target).float().mean()
            val_loss += loss.item()
            val_accuracy += accuracy.item()
            val_n += 1
        val_loss /= val_n
        val_accuracy /= val_n
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.6f}, Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.6f}')

    # Save the model
    torch.save(model.state_dict(), 'bce.pth')