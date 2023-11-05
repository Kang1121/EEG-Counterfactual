import torch.nn as nn
import torchvision.models
import torch


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        resnet = torchvision.models.__dict__["resnet18"](weights=None)
        resnet.conv1 = nn.Conv2d(22, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.fc = nn.Linear(512, num_classes)

        self.resnet = resnet
        self.num_classes = num_classes

    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return {"features": features, "logits": logits}

    def features(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4[0](x)
        return x

    def classifier(self, x):
        x = self.resnet.layer4[1:](x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

    def get_classifier_head(self):
        return nn.Sequential(
            self.resnet.layer4[1:],
            self.resnet.avgpool,
            nn.Flatten(start_dim=1),
            self.resnet.fc,
        )

    def get_features_dim(self):
        return {"n_feat": 2048, "n_row": 7, "n_pixels": 49}

    def get_num_classes(self):
        return self.num_classes