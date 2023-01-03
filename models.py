import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from torchvision.transforms import Normalize


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),
        )

        self.fc_net = nn.Sequential(
            # nn.Linear(512 * 16 * 16, num_classes*4),
            # nn.BatchNorm1d(num_features=num_classes*4),
            # nn.ReLU(inplace=True),

            # nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, num_classes),
        )

    def set_grad(self, val):
        for param in self.conv_net.parameters():
            param.requires_grad = val

    def get_features(self, x, norm=False):
        if norm:
            x = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x)
        features = self.conv_net(x)
        return torch.reshape(features, (features.shape[0], -1))

    def append_last_layer(self, num_classes=2):
        num_out_features = 512 * 1 * 1
        self.out_fc = nn.Linear(num_out_features, num_classes)

    def forward(self, x):
        x = self.conv_net(x)

        # See the CS231 link to understand why this is 16*5*5!
        # This will help you design your own deeper network
        x = x.view(-1, 512 * 1 * 1)
        x = self.fc_net(x)

        # No softmax is needed as the loss function in step 3
        # takes care of that

        return x


class MyResNet(nn.Module):
    def __init__(self, num_classes, pretrain=False):
        super(MyResNet, self).__init__()

        self.resnet18 = models.resnet18(pretrained=pretrain)
        # Replace last fc layer
        self.num_feats = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(self.num_feats, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def set_grad(self, val):
        for param in self.resnet18.parameters():
            param.requires_grad = val

    def get_feature_extractor(self):
        return nn.Sequential(*list(self.resnet18.children())[:-1])

    def get_features(self, x, norm=False):
        if norm:
            x = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x)
        features = self.get_feature_extractor()(x)
        return torch.reshape(features, (features.shape[0], -1))

    def append_last_layer(self, num_classes=2):
        num_out_features = self.num_feats
        self.out_fc = nn.Linear(num_out_features, num_classes)

    def forward(self, x):
        x = self.resnet18(x)
        return x
