import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights
from persim import wasserstein, bottleneck

class TDA_Loss(torch.nn.Module):
    def __init__(self, feature_shape, num_persistence_features=32):
        super(TDA_Loss, self).__init__()
        C,H,W = feature_shape
        self.feature_shape = feature_shape
        self.num_persistence_features = num_persistence_features
        
        self.net = nn.Sequential(
            nn.Conv2d(C, 128, kernel_size=5, stride=1, padding=2), # want -2 on spatial dim, (64, 45, 45) -> (1024, 1, 1) -> (32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1, kernel_size=1),  # reduce to 1 channel
            nn.AdaptiveAvgPool2d((num_persistence_features, num_persistence_features)) # (1024, 1, 1) -> (1, 32, 32)
        )

        self.loss_fn = nn.MSELoss()
        # self.loss_fn = wasserstein()
        # self.loss_fn = bottleneck()

    def forward(self, feature_vector, target_pers_img):
        pred_pers_img = self.net(feature_vector)
        C, H, W = self.feature_shape
        torch.save(pred_pers_img, f"topological_output/pred_pers_img_{C}_{H}_{W}.pt")
        torch.save(target_pers_img, f"topological_output/target_pers_img_{C}_{H}_{W}.pt")
        return self.loss_fn(pred_pers_img, target_pers_img)
        

class Model(torch.nn.Module):
    def __init__(self, num_classes=128, pretrained=False, use_resnet18=True):
        super(Model, self).__init__()

        if use_resnet18:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.resnet_model = resnet18(weights=weights)
            self.feature_dim = 512
        else:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.resnet_model = resnet50(weights=weights)
            self.feature_dim = 2048
        
        '''
        # Modify the first conv layer to accept 4 channels instead of 3
        original_conv1 = self.resnet_model.conv1
        self.resnet_model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None,
        )
        

        # If pretrained, initialize the new conv1 weights from the original ones
        if pretrained:
            with torch.no_grad():
                self.resnet_model.conv1.weight[:, :3] = original_conv1.weight
                self.resnet_model.conv1.weight[:, 3] = original_conv1.weight[:, 0]  # Copy R channel weights to PH
        '''
        self.model = torch.nn.Sequential(*(list(self.resnet_model.children())[:-2]))
        self.seq1 = torch.nn.Sequential(*(list(self.resnet_model.children())[:5]))
        self.seq2 = torch.nn.Sequential(*(list(self.resnet_model.children())[5:6]))
        self.seq3 = torch.nn.Sequential(*(list(self.resnet_model.children())[6:7]))
        self.seq4 = torch.nn.Sequential(*(list(self.resnet_model.children())[7:8]))
        
        self.avgpool = nn.AvgPool2d(kernel_size=6, stride=1, padding=0)

        self.fc1 = nn.Linear(self.feature_dim, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        #x = self.model(x)
        feature1 = self.seq1(x)
        feature2 = self.seq2(feature1)
        feature3 = self.seq3(feature2)
        feature4 = self.seq4(feature3)
        x = self.avgpool(feature4)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x, [feature1, feature2, feature3, feature4]


class LPNet(torch.nn.Module):
    def __init__(self, num_classes=128, pretrained=False, use_resnet18=True):
        super(LPNet, self).__init__()

        if use_resnet18:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.resnet_model = resnet18(weights=weights)
            self.feature_dim = 512
        else:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.resnet_model = resnet50(weights=weights)
            self.feature_dim = 2048

        self.model = torch.nn.Sequential(*(list(self.resnet_model.children())[:-2]))
        self.avgpool = nn.AvgPool2d(kernel_size=6, stride=1, padding=0)

        self.fc1 = nn.Linear(self.feature_dim, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
