from torch import nn
import torch.nn.functional as F
from config import EPSILON
from scipy.spatial.distance import cdist 
import torch
import numpy as np
from scipy.linalg import fractional_matrix_power


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, edges):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edges = edges
        self.weight = np.random.randn(in_features, out_features) * 0.01

    def gcn(self, x):
        I = np.identity(self.edges.shape[0]) #create Identity Matrix of A
        A_hat = self.edges + I #add self-loop to A
        D = np.diag(np.sum(A_hat, axis=0)) #create Degree Matrix of A
        D_half_norm = fractional_matrix_power(D, -0.5) #calculate D to the power of -0.5
        eq = D_half_norm.dot(A_hat).dot(D_half_norm).dot(x).dot(self.weight)
        return eq


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.

    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class Conv4Classifier(nn.Module):
    def __init__(self, k_way: int):
        """Creates a few shot classifier as used in MAML.

        This network should be identical to the one created by `get_few_shot_encoder` but with a
        classification layer on top.

        # Arguments:
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            k_way: Number of classes the model will discriminate between
            final_layer_size: 64 for Omniglot, 1600 for miniImageNet
        """
        super(Conv4Classifier, self).__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        self.fc1 = nn.Linear(1600, k_way)

    def forward(self, x, feature=False):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x4 = x4.view(x.size(0), -1)

        if feature:
            return x4

        return self.fc1(x4)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, topk=10):
        """Channel-wise attention module, it compute the channel weights for 

        # Arguments:
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            k_way: Number of classes the model will discriminate between
            final_layer_size: 64 for Omniglot, 1600 for miniImageNet
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.topk = topk

    def forward(self, x, all=True):
        b, c, _, _ = x.size()
        mask = self.avg_pool(x).view(b, c)
        mask = self.fc(mask)
        weighted_x = x * mask.view(b, c, 1, 1).expand_as(x)

        if all:
            return weighted_x, mask
        else:
            _, idx = mask.topk(self.topk, dim=1)
            return weighted_x[:,idx[0],:,:], mask


class Conv4Attension(nn.Module):
    def __init__(self, k_way: int, sp_k_way: int):
        """Creates a few shot classifier as used in MAML.

        This network should be identical to the one created by `get_few_shot_encoder` but with a
        classification layer on top.

        # Arguments:
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            k_way: Number of classes the model will discriminate between
            k_way: Number of super-classes the model will discriminate between
        """
        super(Conv4Attension, self).__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        self.fc1 = nn.Linear(1600, k_way)
        self.sp_fc = nn.Linear(1600, sp_k_way)
        self.SELayer = SELayer(64)

    def forward(self, x, norm=False):
        x1 = self.conv1(x)  # (batch_size, 64, 42, 42): 64 is attention dimension in attention class
        x2 = self.conv2(x1) # (batch_size, 64, 21, 21): 64 is attention dimension in attention class
        x3 = self.conv3(x2) # (batch_size, 64, 10, 10): 64 is attention dimension in attention class
        x4 = self.conv4(x3) # (batch_size, 64, 5, 5): 64 is attention dimension in attention class
        
        x4_flat = x4.view(x.size(0), -1)
        normalised_x4 = x4_flat / (x4_flat.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        if norm:
            feature = normalised_x4
        else:
            feature = x4_flat


        weighted_x4, _ = self.SELayer(x4)
        # flat and normalize weighted x4:
        weighted_x4_flat = weighted_x4.view(x.size(0), -1)
        # normalised_x4 = weighted_x4_flat / (x.pow(weighted_x4_flat).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        return feature, self.fc1(feature), self.sp_fc(weighted_x4_flat)


class STKH(nn.Module):
    def __init__(self, img_encoder, cat_feature, final_feature, k_way):
        super(STKH, self).__init__()
        self.img_encoder = img_encoder

        self.fc1 = nn.Linear(cat_feature, final_feature)
        self.fc2 = nn.Linear(final_feature, k_way)

    def forward(self, x, kg_features, norm=False):
        img_features, _, sp_outputs = self.img_encoder(x, norm=True)
        combined_features = torch.cat((img_features, kg_features), 1)

        features = self.fc1(combined_features)

        if norm:
            features = features / (features.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        return features, self.fc2(features), sp_outputs

    def predict(self, x, kg_features, norm=False):
        img_features, _, sp_outputs = self.img_encoder(x, norm=True)

        predictions = []

        for i, _ in enumerate(kg_features.shape[0]):
            combined_features = torch.cat((img_features, kg_features[i]), 1)

            features = self.fc1(combined_features)

            if norm:
                features = features / (features.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

            return features, self.fc2(features), sp_outputs
        

class KG_encoder():
    def __init__(self, layer, layer_nums, edges):
        super(KG_encoder, self).__init__()
        self.gcs = []
        for i in range(layer):
            self.gcs.append(GraphConvolution(layer_nums[i], layer_nums[i+1], edges))

    def apply_gc(self, x):
        for gc in self.gcs:
            x = gc.gcn(x)
        return x




# ------------------------------- #
# ResNet
# ------------------------------- #
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, norm=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x) # (batch_size, 64, 56, 56)
        x2 = self.layer2(x1) # (batch_size, 128, 28, 28)
        x3 = self.layer3(x2) # (batch_size, 256, 14, 14)
        x4 = self.layer4(x3) # (batch_size, 512, 7, 7)

        x4 = self.avgpool(x4)
        x4_flat = x4.view(x4.size(0), -1)

        normalised_x4 = x4_flat / (x4_flat.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        if norm:
            feature = normalised_x4
        else:
            feature = x4_flat

        return feature, self.fc(feature)


class ResNetAttention(nn.Module):

    def __init__(self, block, layers, num_classes, num_superclasses, zero_init_residual=False):
        super(ResNetAttention, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.SELayer4 = SELayer(512)
        self.SELayer3 = SELayer(256)
        self.sp_fc = nn.Linear(256 * block.expansion, num_superclasses)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, norm=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x) # (batch_size, 64, 56, 56)
        x2 = self.layer2(x1) # (batch_size, 128, 28, 28)
        x3 = self.layer3(x2) # (batch_size, 256, 14, 14)
        x4 = self.layer4(x3) # (batch_size, 512, 7, 7)

        x4 = self.avgpool(x4)
        x4_flat = x4.view(x4.size(0), -1)

        normalised_x4 = x4_flat / (x4_flat.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        if norm:
            feature = normalised_x4
        else:
            feature = x4_flat


        weighted_x, _ = self.SELayer3(self.avgpool(x3))
        weighted_x_flat = weighted_x.view(x.size(0), -1)
        # import ipdb; ipdb.set_trace()
        # weighted_x4, _ = self.SELayer4(x4)
        # weighted_x4_flat = weighted_x4.view(x.size(0), -1)

        return feature, self.fc(feature), self.sp_fc(weighted_x_flat)


def resnet10(num_classes, num_superclasses):
    """Constructs a ResNet-10 model.
    """
    model = ResNetAttention(BasicBlock, [1, 1, 1, 1], num_classes, num_superclasses)
    return model


def resnet18(num_classes, num_superclasses):
    """Constructs a ResNet-18 model.
    """
    model = ResNetAttention(BasicBlock, [2, 2, 2, 2], num_classes, num_superclasses)
    # model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(num_classes, num_superclasses):
    """Constructs a ResNet-34 model.
    """
    model = ResNetAttention(BasicBlock, [3, 4, 6, 3], num_classes, num_superclasses)
    # model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(num_classes, num_superclasses):
    """Constructs a ResNet-50 model.
    """
    model = ResNetAttention(Bottleneck, [3, 4, 6, 3], num_classes, num_superclasses)
    # model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(num_classes, num_superclasses):
    """Constructs a ResNet-101 model.
    """
    model = ResNetAttention(Bottleneck, [3, 4, 23, 3], num_classes, num_superclasses)
    # model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(num_classes, num_superclasses):
    """Constructs a ResNet-152 model.
    """
    model = ResNetAttention(Bottleneck, [3, 8, 36, 3], num_classes, num_superclasses)
    return model