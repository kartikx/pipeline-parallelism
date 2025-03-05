import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

class NeuralNet(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1), 
        )
        
    def forward(self, x):
        # This forces the first dimension to be the batch, and flattens the rest of the dims.
        x = x.view(x.shape[0], -1) 
        return self.layers(x)

def main():
    train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

    net = NeuralNet(28*28).to(device)
    
    optimizer = optim.Adam(net.parameters(), lr = 0.0001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        net.train()
        for minibatch_x, minibatch_y in train_dataloader:
            minibatch_x, minibatch_y = minibatch_x.to(device), minibatch_y.to(device)

            optimizer.zero_grad()
            output = net(minibatch_x)

            loss = criterion(output, minibatch_y)

            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch} Loss: {loss.item():.4f}")
        

if __name__ == '__main__':
    main()