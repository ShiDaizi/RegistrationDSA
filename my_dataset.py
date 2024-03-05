from PIL import Image
import torch
import cv2
import numpy as np
import os
import pydicom
from torch.utils.data import Dataset


class MyDataSet(Dataset):

    def __init__(self, root: str, image_phase: list, image_name: list, transform = None):
        self.root = root
        self.image_phase = image_phase
        self.image_name = image_name
        self.transform = transform

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, item):
        img = []
        for i in self.image_phase:
            img_path = os.path.join(self.root, i, self.image_name[item])
            ds = pydicom.dcmread(img_path)
            data = np.array(ds.pixel_array)
            data = data - np.min(data)
            data = data / np.max(data)
            data = (data * 255).astype(np.uint8)
            #edge = cv2.Canny(data, 20, 120)
            edge = cv2.Laplacian(data, cv2.CV_64F, ksize=5)
            edge = np.abs(edge).astype(np.uint8)
            if self.transform is not None:
                edge = self.transform(edge)
                data = self.transform(data)
            img.append(data)
            img.append(edge)
        return img


class MNISTDataset(Dataset):
    def __init__(self, image_path: list, transform = None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        mov = np.array(Image.open(self.image_path[item][0])).astype(np.uint8)
        ref = np.array(Image.open(self.image_path[item][1])).astype(np.uint8)
        img = []
        mov = np.clip(mov, 0, 128)
        mov_edge = cv2.Laplacian(mov, cv2.CV_64F, ksize=3)
        mov_edge = np.abs(mov_edge).astype(np.uint8)
        if self.transform is not None:
            mov = self.transform(mov)
            mov_edge = self.transform(mov_edge)
        img.append(mov)
        img.append(mov_edge)

        ref_edge = cv2.Laplacian(ref, cv2.CV_64F, ksize=3)
        ref_edge = np.abs(ref_edge).astype(np.uint8)
        if self.transform is not None:
            ref = self.transform(ref)
            ref_edge = self.transform(ref_edge)
        img.append(ref)
        img.append(ref_edge)
        return img




