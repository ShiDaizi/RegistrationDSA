import os
import json
import torch
import cv2
import pickle
import random
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
DEVICE = 0
def to_cuda(x):
    if USE_CUDA:
        return x.cuda(DEVICE)
    return x

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} dose not exist.".format(root)

    image_phase = [phase for phase in os.listdir(root) if os.path.isdir(os.path.join(root, phase))]
    image_phase.sort()

    indices_phase = dict((k,v) for v,k in enumerate(image_phase))
    json_str = json.dumps(dict((val, key) for key, val in indices_phase.items()), indent=4)
    with open('indices_phase.json', 'w') as json_file:
        json_file.write(json_str)
    supported = [".dcm"]

    val_images = []
    train_images = []

    phase = image_phase[0]
    phase_path = os.path.join(root, phase)
    images = [i for i in os.listdir(phase_path) if os.path.splitext(i)[-1] in supported]
    val_images = random.sample(images, k=int(len(images) * val_rate))
    for img in images:
        if img not in val_images:
            train_images.append(img)

    print(f"{len(train_images)} images for training")
    print(f"{len(val_images)} images for validation")

    return image_phase, train_images, val_images

def split_mnist(root: str, val_rate: float=0.2):
    random.seed(0)
    assert os.path.exists(root), f"dataset root: {root} dose not exist."
    img_doc = [i for i in os.listdir(root) if os.path.isdir(os.path.join(root, i))]
    img_doc.sort()
    img_idx = dict((k,v) for v,k in enumerate(img_doc))
    supported = ['.jpg']

    val_images = []
    train_images = []

    for cla in img_doc:
        cla_path = os.path.join(root, cla)
        ref_images = [os.path.join(cla_path, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
        mov_images = list.copy(ref_images)
        random.shuffle(ref_images)
        images = list(zip(mov_images, ref_images))
        test_images = random.sample(images, k=int(len(images) * val_rate))
        for i in images:
            if i not in test_images:
                train_images.append(i)
            else:
                val_images.append(i)

    print(f"{len(train_images)} images for training")
    print(f"{len(val_images)} images for validation")

    return train_images, val_images



'''
from my_dataset import MNISTDataset
root = "./datarar/mnisttrainset"
train_images, val_images = split_mnist(root)
train_data = MNISTDataset(train_images)
mov, mov_edge, ref, ref_edge = train_data[0]
plt.subplot(141)
plt.imshow(mov, cmap='gray', vmin=0, vmax=255)
plt.subplot(142)
plt.imshow(mov_edge, cmap='gray', vmin=0, vmax=255)
plt.subplot(143)
plt.imshow(ref, cmap='gray', vmin=0, vmax=255)
plt.subplot(144)
plt.imshow(ref_edge, cmap='gray', vmin=0, vmax=255)
plt.show()
'''




'''
from my_dataset import MyDataSet
root = "./datarar/excellent"
image_phase, train_images, val_images = read_split_data(root)
train_data = MyDataSet(root, image_phase, train_images)
img, edge, img2, edge2 = train_data[0]
plt.imshow(img - img2, cmap='gray')
plt.show()
'''