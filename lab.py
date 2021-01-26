"""
This script contains tests that we did on the data
"""
from torch import optim

from dataloader import get_combined_epoch_png
from model import create_nn_v1
import torch

import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary


USE_CUDA = True


def train(model, data_path, epochs=50, batch_size=32):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    results = {}

    for epoch_index in range(epochs):
        data = get_combined_epoch_png(data_path, train=True, batch_size=batch_size)
        for batch_index, (image, label) in enumerate(data):
            if USE_CUDA:
                image, label = image.cuda(), label.cuda()
            optimizer.zero_grad()

            # summary(model, image)
            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

        print(f"[{epoch_index}\{epochs}] | Loss: {loss.item():.4f}")
        results[epoch_index] = {"loss": loss.item()}

    return results


def test(model, data):
    pass


def simple_run(dataset_path):
    model = create_nn_v1()

    model.train()
    results = train(model, "D:/Clouds/YotamDataSet/96x96_32x32", epochs=100)
    print(results)

    model.eval()
    test_set = get_combined_epoch_png(dataset_path, train=False)
    test(model, test_set)
