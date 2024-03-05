import os
import numpy as np
import pydicom
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from pydicom.pixel_data_handlers.util import apply_voi_lut
from smoothTransformer import smoothTransformer2D
import torch
from torchvision import datasets, transforms
# 调用本地的 dicom file


transform = transforms.Compose([transforms.ToTensor()])


def getimg(file_path):
    ds = pydicom.dcmread(file_path)
    img = np.array(ds.pixel_array)
    img = img - np.min(img)
    img = img / np.max(img)
    img = (img * 255).astype(np.uint8)
    return img

def create_deformed_image(root: str, save: str):
    assert os.path.exists(root), f"image root: {root} dose not exist."
    supported = ['.jpg']
    img = [i for i in os.listdir(root) if os.path.splitext(i)[-1] in supported]
    for i in os.listdir(root):
        if i not in img:
            continue
        data = cv2.imread(os.path.join(root, i))
        theta = np.random.rand(1)[0]
        x = np.random.rand(1)[0]
        y = np.random.rand(1)[0]
        M1 = np.float32([[1, 0, x], [0, 1, y]])
        cols, rows, _ = data.shape
        M2 = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
        data = cv2.warpAffine(data, M2, (cols, rows))
        data = cv2.warpAffine(data, M1, (cols, rows))
        cv2.imwrite(os.path.join(save, i), data)

root = "./datarar/original/2"
save = "./datarar/deformed/2"
create_deformed_image(root, save)

'''
create_deformed_image(root, save)
folder_path1 = r".\datarar\excellent\0"
file_name1 = "50.dcm"
file_path1 = os.path.join(folder_path1, file_name1)
data1 = getimg(file_path1)



folder_path2 = r".\datarar\excellent\1"
file_name2 = "50.dcm"
file_path2 = os.path.join(folder_path2, file_name2)
data2 = getimg(file_path2)

data3 = data1 - data2
#data3 = np.array(pydicom.dcmread(file_path1).pixel_array) - np.array(pydicom.dcmread(file_path2).pixel_array)
print(data3)
data3 = data3 - np.min(data3)
data3 = data3 / np.max(data3)
data3 = (data3 * 255).astype(np.uint8)
#plt.imshow(data1, cmap="gray")
#plt.show()

#data1.dtype=np.uint8
#data2.dtype=np.uint8
assert data1 is not None, "file could not be read, check with os.path.exists()"

lap1 = cv2.Laplacian(data1,cv2.CV_16S,ksize = 7)
lap2 = cv2.Laplacian(data2,cv2.CV_16S,ksize = 7)
edges1 = cv2.Canny(data1,20,120)
edges2 = cv2.Canny(data2,20,120)
plt.subplot(131),plt.imshow(lap1,cmap = 'gray')
plt.title('Original 0 Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(lap2,cmap = 'gray')
plt.title('Edge1 Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(lap2-lap1,cmap = 'gray')
plt.title('Edge2 Image'), plt.xticks([]), plt.yticks([])
plt.show()
np.savetxt('output.txt', data1, fmt="%d", delimiter=',')
'''
