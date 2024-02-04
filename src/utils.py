import json
import os
from copy import deepcopy
from random import sample, choices

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from Custom_CNN import Custom_CNN
from CNN import CNN


def aux_info(dataset_name, model_name):
    # Determine the number of classes and number of channels for the desired dataset
    num_classes, num_channels = None, None
    if dataset_name in ["MNIST", "FMNIST"]:
        num_classes = 10
        num_channels = 1
    elif dataset_name == "CIFAR10":
        num_classes = 10
        num_channels = 3
    elif dataset_name == "FEMNIST":
        num_classes = 62
        num_channels = 1

    # 1) Determine the appropriate pre-processing transform for the desired dataset
    transform = None
    if dataset_name in ["MNIST", "FMNIST"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_name == "FEMNIST":
        transform = transforms.ToTensor()

    # 2) Adjust the transform based on the model that is going to be used
    if model_name == "SVM":
        transform = transforms.Compose([
            transform,
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
    elif model_name == "CNN":
        transform = transforms.Compose([
            transform,
            transforms.Resize((32, 32))
        ])

    return num_classes, transform, num_channels


def dataset_info(dataset_name, transform):
    train_set, test_set = None, None

    if dataset_name == "MNIST":
        train_set = datasets.MNIST('../data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST('../data', train=False, download=True, transform=transform)

    elif dataset_name == "FMNIST":
        train_set = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST('../data', train=False, download=True, transform=transform)

    elif dataset_name == "CIFAR10":
        train_set = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10('../data', train=False, download=True, transform=transform)

    elif dataset_name == "FEMNIST":
        train_set = json_to_data(os.path.join(os.getcwd(), "../data/leaf/data/femnist/data/train"), transform)
        test_set = json_to_data(os.path.join(os.getcwd(), "../data/leaf/data/femnist/data/test"), transform)

        train_set = sample(train_set, int(0.1 * len(train_set)))
        test_set = sample(test_set, int(0.1 * len(test_set)))

    input_dim = calculate_input_dim(train_set[0][0].shape)
    return list(train_set), list(test_set), input_dim


def model_info(model_name, input_dim, num_classes, num_channels):
    model, criterion = None, None
    if model_name == "SVM":
        model = torch.nn.Linear(input_dim, num_classes)
        criterion = torch.nn.MultiMarginLoss()
    elif model_name == "CNN":
        model = CNN(num_classes, num_channels)
        criterion = torch.nn.CrossEntropyLoss()
    elif model_name == "Custom_CNN":
        model = Custom_CNN(num_classes, num_channels)
        criterion = torch.nn.CrossEntropyLoss()
    elif model_name == "VGG11":
        model = models.vgg11(weights='DEFAULT')
        criterion = torch.nn.CrossEntropyLoss()

    model_dim = calculate_model_dim(model.parameters())
    return model, criterion, model_dim


def calculate_input_dim(shape):
    dim = 1
    for ax in shape:
        dim *= ax
    return dim


def calculate_model_dim(model_params):
    model_dim = 0
    for param in model_params:
        model_dim += len(param.flatten())
    return model_dim


def calculate_accuracy(model, test_set):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    test_loader = DataLoader(
        test_set,
        batch_size=32,
        shuffle=False
    )

    correct = 0
    for dataX, dataY in iter(test_loader):
        dataX, dataY = dataX.to(device), dataY.to(device)
        output = model(dataX)
        pred = output.argmax(dim=1)
        correct += torch.tensor(pred == dataY).int().sum().item()

    return correct / len(test_set)


def moving_average(x, window=32):
    if len(x) <= window or window == 1:
        return x
    y = [0 for _ in range(len(x) - window + 1)]

    y[0] = sum(x[:window]) / window
    for i in range(len(y) - 1):
        y[i + 1] = y[i] + (x[i + window] - x[i]) / window
    return y


def moving_average_df(df, window=32):
    df_ret = pd.DataFrame()
    for column_name in df:
        ret_col = moving_average(df[column_name], window)
        df_ret[column_name] = ret_col
    return df_ret


def generate_train_sets(train_set, num_agents, num_classes, labels_per_agent):
    # First shuffle the training set,
    # and then separate it to a dictionary where each entry only contains data coming from one class
    shuffled = sample(train_set, k=len(train_set))
    separated_by_output = {j: [data for data in shuffled if data[1] == j] for j in range(num_classes)}

    # Determine number of data splits from each class for each agent
    total_data_splits_count = num_agents * labels_per_agent
    data_splits_per_class = total_data_splits_count // num_classes
    rem_classes = total_data_splits_count % num_classes
    each_class_div = [data_splits_per_class for _ in range(total_data_splits_count - rem_classes)]
    each_class_div.extend([data_splits_per_class + 1 for _ in range(rem_classes)])

    each_class_div = sample(each_class_div, k=len(each_class_div))
    available_splits = {j: each_class_div[j] for j in range(num_classes)}

    data_splits = {j: [] for j in range(num_classes)}
    for j in range(num_classes):
        div = len(separated_by_output[j]) // (each_class_div[j])
        data_splits[j].extend([separated_by_output[j][i * div: (i + 1) * div] for i in range(each_class_div[j] - 1)])
        data_splits[j].append(separated_by_output[j][(each_class_div[j] - 1) * div: len(separated_by_output[j])])

    separated = [[] for _ in range(num_agents)]
    for i in range(num_agents):
        available_splits_temp = deepcopy(available_splits)
        chosen_splits = []
        for j in range(labels_per_agent):
            chosen_splits.extend(choices(list(available_splits_temp.keys()),
                                         weights=list(available_splits_temp.values()), k=1))
            del available_splits_temp[chosen_splits[-1]]

        for j in chosen_splits:
            separated[i].extend(data_splits[j][0])
            available_splits[j] -= 1

    separated_shuffled = [sample(separated[i], k=len(separated[i])) for i in range(len(separated))]
    return separated_shuffled


def json_to_data(dirname, transform):
    data = []
    for json_file in os.listdir(dirname):
        with open(os.path.join(dirname, json_file), "r") as f:
            for _, subset in json.load(f)["user_data"].items():
                dim = int(np.sqrt(len(subset['x'][0])))
                data.extend(
                    [(transform(np.reshape(subset['x'][i], (1, dim, dim)).T.astype(np.float32)), subset['y'][i])
                     for i in range(len(subset['x']))])
    return data


def save_results(log, filepath):
    sheets = ['iters', 'iters_sampled']
    with pd.ExcelWriter(filepath) as writer:
        for i in range(2):
            df_i = pd.DataFrame(log[i])
            df_i.to_excel(writer, sheet_name=sheets[i])


def determine_DandB(DandB, initial_prob_sgds, initial_prob_aggrs):
    D, B = DandB
    if D is None:
        D = int(np.mean([1/ips for ips in initial_prob_sgds]))
    if B is None:
        B = int(np.mean([[1/ipa for ipa in row] for row in initial_prob_aggrs]))
    return (D, B)
