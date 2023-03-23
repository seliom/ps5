import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import optim
import pickle
from cnn import MNISTConvNet
import time

# For reproducability
torch.manual_seed(0)

trainset = MNIST('.', train=True, download=True, 
                      transform=ToTensor())
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

lr = 1e-3
num_epochs = 500

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Running on " + dev)

model = MNISTConvNet().to(dev)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

def train():
    for epochs in range(num_epochs):
        running_loss = 0.0
        num_correct = 0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs.to(dev))
            loss = loss_fn(outputs, labels.to(dev))
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            _, idx = outputs.max(dim=1)
            num_correct += (idx == labels.to(dev)).sum().item()
        print('Loss: {} Accuracy: {}'.format(running_loss/len(trainloader),
            num_correct/len(trainloader)))

    # Save the trained model
    pickle_file = 'mnist_cnn_model_500.pkl'

    with open(pickle_file, 'wb') as f:
        pickle.dump(model.state_dict(), f)

    print(f"Model saved to {pickle_file}")



start = time.time()
train()
end = time.time()
print('GPU time: %.2f seconds' % (end - start))

# dev = "cpu"
# start = time.time()
# train()
# end = time.time()
# print('CPU time: %.2f seconds' % (end - start))

