import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pickle
from cnn import MNISTConvNet
from sklearn.metrics import confusion_matrix

# Load the saved model
pickle_file = 'mnist_cnn_model_500.pkl'
with open(pickle_file, 'rb') as f:
    state_dict = pickle.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MNISTConvNet()
model.load_state_dict(state_dict)
model.to(device)

testset = MNIST('.', train=False, download=True, transform=ToTensor())
testloader = DataLoader(testset, batch_size=64, shuffle=False)

model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"Accuracy of the CNN model on the test set: {accuracy}%")

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)
