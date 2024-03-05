import net
import utils as U
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as StepLR
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from my_dataset import MyDataSet
from my_dataset import MNISTDataset
from smoothTransformer import smoothTransformer2D
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import cv2 as cv
import kornia.morphology as mm


def minmaxscaler(data):
    min = data.min()
    max = data.max()
    return (data - min)/(max-min)


def read_dcm(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array
    img = img - np.min(img)
    img = img / np.max(img)
    img = (img * 255).astype(np.uint8)
    mov = img[2]
    #ref = img[40]
    ref = img[12]

    #Canny
    '''
    refedge = transform(cv.Canny(ref, 40, 60))
    movedge = transform(cv.Canny(mov, 40, 60))
    refedge = torch.unsqueeze(refedge, 0)
    movedge = torch.unsqueeze(movedge, 0)   
    '''


    '''
    plt.subplot(121), plt.imshow(ref, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(refedge, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    '''

    ref = transform(ref)
    mov = transform(mov)

    ref = torch.unsqueeze(ref, 0)
    mov = torch.unsqueeze(mov, 0)

    #morphology
    #gauss_ker = torch.tensor([[[[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]]], dtype=torch.float32)
    gauss_ker = torch.tensor([[[[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]]]], dtype=torch.float32) / 273
    #lap_ker = torch.tensor([[[[-1, -1, -1], [-1, 4, -1], [0, -1, 0]]]], dtype=torch.float32)
    lap_ker = torch.tensor([[[[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]]], dtype=torch.float32)
    # lap_ker = lap_ker.repeat(1, 1, 1, 1)
    refedge = F.conv2d(ref, gauss_ker, padding=0, groups=1)
    movedge = F.conv2d(mov, gauss_ker, padding=0, groups=1)

    kernel = torch.zeros(3, 3)
    M = mm.opening(mm.closing(mov, kernel), kernel)
    movedge = mm.dilation(mm.closing(M, kernel), kernel) - mm.closing(M, kernel)
    movedge = minmaxscaler(movedge)
    M = mm.opening(mm.closing(ref, kernel), kernel)
    refedge = mm.dilation(mm.closing(M, kernel), kernel) - mm.closing(M, kernel)
    refedge = minmaxscaler(refedge)    



    #Laplacian
    '''
    #gauss_ker = torch.tensor([[[[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]]], dtype=torch.float32)
    gauss_ker = torch.tensor([[[[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]]]], dtype=torch.float32) / 273
    #lap_ker = torch.tensor([[[[-1, -1, -1], [-1, 4, -1], [0, -1, 0]]]], dtype=torch.float32)
    lap_ker = torch.tensor([[[[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]]], dtype=torch.float32)
    # lap_ker = lap_ker.repeat(1, 1, 1, 1)
    refedge = F.conv2d(ref, gauss_ker, padding=0, groups=1)
    movedge = F.conv2d(mov, gauss_ker, padding=0, groups=1)
    refedge = F.conv2d(refedge, lap_ker, padding=0, groups=1).abs()
    movedge = F.conv2d(movedge, lap_ker, padding=0, groups=1).abs()
    refedge = minmaxscaler(refedge)
    movedge = minmaxscaler(movedge)    
    '''


    return ref, mov, refedge, movedge


epochs = 500
lr = 0.0001
c = 4
lambd = 0.01
img_ch = 1
mod_epoch = 10
save_folder = './save'
transform = transforms.Compose([
    transforms.ToTensor()
])

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print('Save folder created')



path_dcm = './head.dcm'
ref, mov, refedge, movedge = read_dcm(path_dcm)

model = U.to_cuda(net.Net(2, c))
optimizer = optim.Adam(model.parameters(), lr=lr)

ref = U.to_cuda(ref)
mov = U.to_cuda(mov)
refedge = U.to_cuda(refedge)
movedge = U.to_cuda(movedge)
ref = transforms.CenterCrop(724)(ref)
mov = transforms.CenterCrop(724)(mov)
refedge = transforms.CenterCrop(724)(refedge)
movedge = transforms.CenterCrop(724)(movedge)

#plt.subplot(121), plt.imshow(movedge.squeeze().cpu().numpy(), cmap='gray')
#plt.subplot(122), plt.imshow(refedge.squeeze().cpu().numpy(), cmap='gray')
#plt.show()

mode = '12'

if mode == '1':
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        deformable, moved, sgrid = model(mov, ref)
        movededge, _ = smoothTransformer2D(movedge, deformable, c)

        ref_loss = torch.sum(((movededge - refedge) * (1.0 - refedge)) ** 2)
        mov_loss = torch.sum(((moved - ref) * (refedge) ) ** 2)
        mse_loss = ref_loss + mov_loss
        #mse_loss = mov_loss
        def_loss = lambd * torch.sum(torch.abs(deformable))
        loss = mse_loss + def_loss

        loss.backward()
        optimizer.step()

        if (epoch+1) % mod_epoch == 0:
            print(f"Epoch: {epoch+1}/{epochs} Loss: {loss}")
    torch.save(model.state_dict(), f"{save_folder}/model_{epochs}.pt")


elif mode == '12':
    model_root = os.path.join(save_folder, f"model_{epochs}.pt")
    model.load_state_dict(torch.load(model_root))
    model.eval()
    deformable, moved, sgrid = model(mov, ref)
    #movededge, _ = smoothTransformer2D(movedge, deformable, c)

    #print(ref.shape)
    ref = ref.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    mov = mov.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    moved = moved.squeeze(0).permute(1, 2, 0).data.cpu().numpy()
    sgrid = sgrid.squeeze(0).data.cpu().numpy()

    xx, yy = np.meshgrid(range(mov.shape[0]), range(mov.shape[1]))
    dx, dy = sgrid[0, :, :] + xx, sgrid[1, :, :] + yy

    plt.figure(figsize=(10, 4))
    plt.subplot(2, 3, 1)
    plt.imshow(mov, cmap='gray', vmin=0, vmax=1)
    plt.title('Mov Image')
    plt.subplot(2, 3, 2)
    plt.imshow(ref, cmap='gray', vmin=0, vmax=1)
    plt.title('Ref Image')
    plt.subplot(2, 3, 3)
    plt.imshow(moved, cmap='gray', vmin=0, vmax=1)
    plt.contour(dx, 50, alpha=0.5, linewidths=0.5)
    plt.contour(dy, 50, alpha=0.5, linewidths=0.5)
    plt.title('Moved Image')
    plt.subplot(2, 3, 4)
    sub = minmaxscaler(ref - moved)
    sub = sub[20:-40, 20:-40]
    plt.imshow(sub, cmap='gray')
    plt.title('Subtraction Image')
    plt.subplot(2, 3, 5)
    osub = minmaxscaler(ref - mov)
    osub = osub[20:-40, 20:-40]
    plt.imshow(osub, cmap='gray')
    plt.title('Original Subtraction Image')
    plt.tight_layout()
    plt.savefig('example-2d-output.png')
    plt.show()














