"""
This script contains tests that we did on the data
"""
from dataloader import get_combined_png
from model import create_nn_v1
import torch

import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary


def train(model, data, size=1000, batch=32):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # test_results =
    for i, (image, label) in enumerate(data):
        image, label = image.cuda(), label.cuda()
        # summary(model, image)
        outputs = model(image)
        loss = criterion(outputs, label)
        print(f"[{(i+1) * batch}\{size}] | Loss: {loss.item():.4f} | "
              f"label {list(map(float, label[0]))} | {list(map(float, outputs[0]))}")
        loss.backward()
        optimizer.step()

        if i*batch > size:
            break


def test(model, data):
    pass


def simple_run(dataset_path):
    model = create_nn_v1()

    model.train()
    train_set = get_combined_png(dataset_path, train=True)
    train(model, train_set)

    model.eval()
    test_set = get_combined_png(dataset_path, train=False)
    test(model, test_set)
