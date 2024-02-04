import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, n_classes, n_channels):
        super(CNN, self).__init__()

        # convolutional layer
        self.conv1 = nn.Conv2d(n_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # fully connected layers
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, n_classes)

        # dropout
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))

        # flattening
        x = x.view(-1, 120)

        # fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return F.log_softmax(x, -1)
