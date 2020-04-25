import os, sys

import torch

import torchvision
from torchvision import transforms
from torchvision import datasets
from cifar import CIFAR10

from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, RandomContrast, RandomGamma, RandomBrightness, HueSaturationValue, ShiftScaleRotate
from albumentations.pytorch import ToTensor

bUseAlbumentations = True
    
def get_data_loader():
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5))])

    if bUseAlbumentations:
        train_transform = Compose([
                HorizontalFlip(p=0.5),
                #RandomContrast(limit=0.2, p=0.5),
                #RandomGamma(gamma_limit=(80, 120), p=0.5),
                #RandomBrightness(limit=0.2, p=0.5),
                #HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
                #                   val_shift_limit=10, p=.9),
                # CLAHE(p=1.0, clip_limit=2.0),
                #ShiftScaleRotate(
                #    shift_limit=0.0625, scale_limit=0.1, 
                #    rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
                
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensor()
                ])
        
        test_transform = Compose([
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensor()
                ])
    
    SEED = 1
    torch.manual_seed(SEED)

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(SEED)

    dataloader_args = dict(num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    trainset = CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=4)

    testset = CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=8,
                                             shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes