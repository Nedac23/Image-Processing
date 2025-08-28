import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
import cv2

from scipy.stats.kde import gaussian_kde

import os
from math import log10, sqrt 
import pywt


#https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr, mse 




opt = 0
if(opt == 0):
    threshold = 'soft'
elif(opt == 1):
    threshold = 'hard'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#https://www.geeksforgeeks.org/denoising-autoencoders-in-machine-learning/
transform = transforms.Compose([
    transforms.Resize((28, 28)),   
    transforms.Grayscale(num_output_channels=1), 
    transforms.ToTensor(),           
])
mnist_dataset_train = datasets.CIFAR10(
    root='./data',  train=True, download=True, transform=transform)
mnist_dataset_test = datasets.CIFAR10(
     root='./data',  train=False, download=True, transform=transform)
batch_size = 128
train_loader = torch.utils.data.DataLoader(
    mnist_dataset_train, batch_size=batch_size, shuffle=True)
imagepath = 'path for image'
data1 = Image.open(imagepath)
resizetransfrom = transforms.Resize((28, 28))
tensor_data = transform(resizetransfrom(data1))
test_loader = torch.utils.data.DataLoader(
    mnist_dataset_test, batch_size=1, shuffle=False)

class DAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        out = self.relu(self.fc3(h2))
        if not self.training:
            out_np = out.detach().cpu().numpy()
            for i in range(out_np.shape[0]):
                out_np[i] = pywt.threshold(out_np[i], 0.5, mode=threshold)
            out = torch.from_numpy(out_np).to(out.device).float()
        return out
    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        out = self.fc6(h5)
        if not self.training:
            out_np = out.detach().cpu().numpy()
            for i in range(out_np.shape[0]):
                out_np[i] = pywt.threshold(out_np[i], 0.5, mode=threshold)
            out = torch.from_numpy(out_np).to(out.device).float()
        return self.sigmoid(out)
    def forward(self, x):
        q = self.encode(x.view(-1, 784))
        return self.decode(q)
def train(epoch, model, train_loader, optimizer,  cuda=True):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data.to(device)
        optimizer.zero_grad()
        data_noise = torch.randn(data.shape).to(device)
        data_noise = data + data_noise
        recon_batch = model(data_noise.to(device))
        loss = criterion(recon_batch, data.view(data.size(0), -1).to(device))
        loss.backward()
        train_loss += loss.item() * len(data)
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                                                                           100. * batch_idx /
                                                                           len(train_loader),
                                                                           loss.item()))
    print('====&gt; Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
epochs = 10
model = DAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.MSELoss()
for epoch in range(1, epochs + 1):
    train(epoch, model, train_loader, optimizer, True)
tensor_data = tensor_data.to(device)
optimizer.zero_grad()
data_noise = 0.1 * torch.randn(tensor_data.shape).to(device)
data_noise = tensor_data + data_noise
recon_batch = model(data_noise.to(device))
plt.figure(figsize=(20, 12))
for i in range(1):
    plt.subplot(3, 5, 1+i)
    plt.imshow(data_noise[i, :, :].view(
        28, 28).detach().numpy(), cmap='binary')
    plt.subplot(3, 5, 6+i)
    plt.imshow(recon_batch[i, :].view(28, 28).detach().numpy(), cmap='binary')
    plt.axis('off')
    plt.subplot(3, 5, 11+i)
    plt.imshow(tensor_data[i, :, :].view(28, 28).detach().numpy(), cmap='binary')
    plt.axis('off')
plt.show()
value, mse = PSNR(tensor_data[i, :, :].view(28, 28).detach().numpy(), recon_batch[i, :].view(28, 28).detach().numpy()) 
print(f"PSNR value is {value} dB") 
print(f"MSE value is {mse} dB") 
