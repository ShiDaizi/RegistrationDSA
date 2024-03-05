import net
import utils as U
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pylab as plt
import numpy as np
import os
from my_dataset import MyDataSet


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--model_id', type=int, default=0, help='specific model that you want to load')
parser.add_argument('--c', type=int, default=2, help='displacement')
parser.add_argument('--img_channels', type=int, default=2, help='channels of images')
parser.add_argument('--save_folder', type=str, default='./models', help='where to save the trained models')
args = parser.parse_args()

model = U.to_cuda(net.Net(ch_in=args.img_channels, c=args.c))
model_root = os.path.join(args.save_folder, f'model_{args.model_id}.pt')
model.load_state_dict(torch.load(model_root))

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




#excellent
'''
root = "./datarar/excellent"
image_phase, train_images, val_images = U.read_split_data(root)
train_dataset = MyDataSet(root, image_phase, train_images, transform=transform)
test_dataset = MyDataSet(root, image_phase, val_images, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)
'''


img = next(iter(test_loader))
img[0] = U.to_cuda(img[0])
img[1] = U.to_cuda(img[1])
img[2] = U.to_cuda(img[2])
img[3] = U.to_cuda(img[3])
deformable, deformed, sgrid = model(img[0], img[2])

mov_inp = img[0].permute(0, 2, 3, 1)
ref_inp = img[2].permute(0, 2, 3, 1)
mov_edge = img[1].permute(0, 2, 3, 1)
ref_edge = img[3].permute(0, 2, 3, 1)
deformed = deformed.permute(0, 2, 3, 1)

mov_inp = mov_inp.data.cpu().numpy()
ref_inp = ref_inp.data.cpu().numpy()
mov_edge = mov_edge.data.cpu().numpy()
ref_edge = ref_edge.data.cpu().numpy()
deformed, sgrid = deformed.data.cpu().numpy(), sgrid.data.cpu().numpy()

xx, yy = np.meshgrid(range(mov_inp.shape[1]), range(mov_inp.shape[2]))
dx, dy = np.squeeze(sgrid[:, 0, :, :]) + xx, np.squeeze(sgrid[:, 1, :, :]) + yy

plt.figure(figsize=(10, 4))
plt.subplot(2, 3, 1)
plt.imshow(np.squeeze(mov_inp), cmap='gray', vmin=0, vmax=1)
plt.title('Moving Image')
plt.subplot(2, 3, 2)
plt.imshow(np.squeeze(ref_inp), cmap='gray', vmin=0, vmax=1)
plt.title('Reference Image')
plt.subplot(2, 3, 3)
plt.imshow(np.squeeze(deformed), cmap='gray', vmin=0, vmax=1)
plt.contour(dx, 50, alpha=0.5, linewidths=0.5)
plt.contour(dy, 50, alpha=0.5, linewidths=0.5)
plt.title('Moved Image')
plt.subplot(2, 3, 4)
sub = np.squeeze(ref_inp - deformed)
#sub -= np.min(sub)
#sub = sub / np.max(sub)
#sub = (sub * 255).astype(np.uint8)
plt.imshow(sub, cmap='gray')
plt.title('Subtraction Image')
plt.subplot(2, 3, 5)
osub = np.squeeze(ref_inp - mov_inp)
#osub -= np.min(osub)
#osub = osub / np.max(osub)
#osub = (osub * 255).astype(np.uint8)
plt.imshow(osub, cmap='gray')

plt.title('Original Subtraction Image')
#plt.subplot(2, 3, 6)
#plt.imshow(np.squeeze(ref_edge), cmap='gray', vmin=0, vmax=1)
#plt.title('Reference Edge Image')
plt.tight_layout()
plt.savefig('example-2d-output.png')
plt.show()





