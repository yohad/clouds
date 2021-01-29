"""
This script contains tests that we did on the data
"""
from collections import namedtuple

from torch import optim

from dataloader import get_combined_epoch_png
from model import create_nn_v1
import torch
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary


USE_CUDA = True

class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def __str__(self):
        return f"{self.correct}/{self.total} "\
               f"[{self.correct/self.total}]"

def _calculate_accuracy(predictions, labels):
    """
    Get a prediction tensor BATCHx5 and compares with BATCHx1 class labels
    """
    batches, _ = predictions.shape
    total, correct = 0, 0
    for i in range(batches):
        predicted_label = predictions[i].argmax()
        if predicted_label == labels[i]:
            correct += 1

        total += 1

    return correct, total


def train(model, data_path, epochs=50, batch_size=32):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    results = {}

    for epoch_index in range(epochs):
        accuracy = Accuracy()
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

            # save important data for later
            correct, total = _calculate_accuracy(outputs, label)
            accuracy.correct += correct
            accuracy.total += total

        print(f"[{epoch_index}\{epochs}] | Loss: {loss.item():.4f} | "
              f"accuracy: {accuracy}]")
        results[epoch_index] = {
            "loss": loss.item(),
            "accuracy": accuracy
        }

    return results


def test(model, data_path, batch_size=32):
    accuracy = Accuracy()
    data = get_combined_epoch_png(data_path, train=False, batch_size=batch_size)
    for batch_index, (image, label) in enumerate(data):
        if USE_CUDA:
            image, label = image.cuda(), label.cuda()
        outputs = model(image)

        correct, total = _calculate_accuracy(outputs, label)
        accuracy.correct += correct
        accuracy.total += total

    print(f"Test accuracy is {accuracy}]")
    return accuracy


def simple_run(dataset_path):
    model = create_nn_v1()

    model.train()
    results = train(model, "D:/Clouds/YotamDataSet/96x96_32x32", epochs=50)
    print(results)

    model.eval()
    results = test(model, "D:/Clouds/YotamDataSet/96x96_32x32")
