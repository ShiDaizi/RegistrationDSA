import net
import utils as U
from torch.utils.data import DataLoader
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pylab as plt
import shutil
import os
import argparse
from my_dataset import MyDataSet
from smoothTransformer import smoothTransformer2D

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='batchsize')
parser.add_argument('--lr', type=float, default=1e-3, help='learning_rate')
parser.add_argument('--c',type=int, default=2, help='displacement')
parser.add_argument('--lambd', type=float, default=1e-7, help='value of the regularizer applied on the spatial gradients')
parser.add_argument('--img_channels', type=int, default=2, help='channels of images')
parser.add_argument('--save_folder', type=str, default='./models', help='root to save the model')
parser.add_argument('--epochs', type=int, default=1, help='training epochs')
parser.add_argument('--mod_epoch', type=int, default=25)
args = parser.parse_args()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(64, 64))
])

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
])

#mnist dataset
from my_dataset import MNISTDataset
root = "./datarar/mnisttrainset"
train_images, val_images = U.split_mnist(root)
train_dataset = MNISTDataset(train_images, transform=mnist_transform)
test_dataset = MNISTDataset(val_images, transform=mnist_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)



#excellent dataset
'''
root = "./datarar/excellent"
image_phase, train_images, val_images = U.read_split_data(root)
train_dataset = MyDataSet(root, image_phase, train_images, transform=transform)
test_dataset = MyDataSet(root, image_phase, val_images, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)
'''

#Random MNIST dataset
'''
train_dataset = datasets.MNIST('./mnist_data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./mnist_data', train=False, download=False, transform=transform)

mov_train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
mov_test_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

ref_train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
ref_test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)
'''

model = U.to_cuda(net.Net(ch_in=args.img_channels, c=args.c))
criterion = U.to_cuda(torch.nn.MSELoss())
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if os.path.exists(args.save_folder):
    shutil.rmtree(args.save_folder)
os.mkdir(args.save_folder)

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for idx, img in enumerate(train_loader):
        img[0] = U.to_cuda(img[0])
        img[1] = U.to_cuda(img[1])
        img[2] = U.to_cuda(img[2])
        img[3] = U.to_cuda(img[3])

        optimizer.zero_grad()

        deformable, deformed, sgrid = model(img[0], img[2])

        im, _ = smoothTransformer2D(img[1], deformable, args.c)
        ref_loss = torch.sum(((im - img[3]) * (1.0 - img[3])) ** 2)
        mov_loss = torch.sum(((deformed - img[2])*img[3])**2) #+ 0.1*torch.sum((1-deformed)**2)
        mse_loss = ref_loss + mov_loss
        def_loss = args.lambd * torch.sum(torch.abs(deformable))
        loss = mse_loss + def_loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if idx % args.mod_epoch == 0:
            print('Train (Epoch {}) [{}/{}]\tLoss: {:.4f}'.format(
                epoch, idx, int(len(train_loader.dataset) / args.batch_size), loss.item()/img[0].shape[0]))


    print('\nTrain set: Average loss: {:.4f}'.format(total_loss / len(train_loader.dataset)))

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, img in enumerate(test_loader):
            img[0] = U.to_cuda(img[0])
            img[1] = U.to_cuda(img[1])
            img[2] = U.to_cuda(img[2])
            img[3] = U.to_cuda(img[3])

            deformable, deformed, sgrid = model(img[0], img[2])

            im, _ = smoothTransformer2D(img[1], deformable, args.c)
            ref_loss = torch.sum(((im - img[3]) * (1.0 - img[3])) ** 2)
            mov_loss = torch.sum(((deformed - img[2]) * img[3]) ** 2) #+ 0.1 * torch.sum((1 - deformed) ** 2)
            mse_loss = ref_loss + mov_loss
            def_loss = args.lambd * torch.sum(torch.abs(deformable))
            test_loss += mse_loss + def_loss
        print('Test set: Average loss: {:.4f}\n'.format(
            test_loss/len(test_loader.dataset)))

    torch.save(model.state_dict(), "{}/model_{}.pt".format(args.save_folder, epoch))