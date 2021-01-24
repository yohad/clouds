"""
This module contains different models for clouds classifications
"""
import torch


def _get_device():
    """
    Get a device to run a model on. GPU is preferred
    """
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    return device


def create_nn_v1():
    """
    First model of the NN
    [C2D, B, P]x3 -> 2048 -> 100 -> SoftMax
    """
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=(3, 3)),
        torch.nn.BatchNorm2d(32),
        torch.nn.MaxPool2d(3),
        torch.nn.ReLU(),

        torch.nn.Conv2d(32, 32, kernel_size=(3, 3)),
        torch.nn.BatchNorm2d(32),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),

        torch.nn.Conv2d(32, 32, kernel_size=(3, 3)),
        torch.nn.BatchNorm2d(32),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),

        torch.nn.Flatten(),

        torch.nn.Linear(1152, 100),
        torch.nn.Linear(100, 5),
        torch.nn.Softmax(1)
    )
    return model.to(_get_device())
