import torch.nn as nn
from .ChannelAtt import SELayer
import torch.nn.functional as F

__all__ = ['resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152','resnet10_att', 'resnet18_att', 'resnet34_att', 'resnet50_att']


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

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, remove_linear=False):
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
        if remove_linear:
            self.fc = None
        else:
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

    def forward(self, x, feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.fc is None:
            if feature:
                return x, None
            else:
                return x
        if feature:
            x1 = self.fc(x)
            return x, x1
        x = self.fc(x)
        return x


class ResNetAttention(nn.Module):

    def __init__(self, block, layers, num_classes, sp_embedding_feature_dim, zero_init_residual=False, pool_type='avg_pool', top_k=16):
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

        self.top_k = top_k
        self.dim_feature = 256 * block.expansion

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if pool_type == 'avg_pool':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool_type == 'max_pool':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))

        # ------------------------------- #
        # Version 3
        # ------------------------------- #
        # layer 3 attention 
        # self.SELayer = SELayer(256, self.top_k)
        # self.sp_fc = nn.Linear(self.top_k * 21 * 21, sp_embedding_feature_dim)

        # layer 4 attention
        # self.SELayer = SELayer(512, self.top_k)
        # self.sp_fc = nn.Sequential(nn.Dropout(0.2),
        #                             nn.Linear(self.top_k * 11 * 11, sp_embedding_feature_dim))

        # ------------------------------------------------------------------- #
        # Version 4: Apply conv on attented features
        # ------------------------------------------------------------------- #
        # self.sp_layer = nn.Sequential(conv3x3(256, 512, 2),
        #                             nn.BatchNorm2d(512),
        #                             SELayer(512, topk=-1),
        #                             nn.ReLU(inplace=True),
        #                             conv3x3(512, sp_embedding_feature_dim, 2),
        #                             nn.BatchNorm2d(sp_embedding_feature_dim),
        #                             SELayer(sp_embedding_feature_dim, topk=-1),
        #                             nn.ReLU(inplace=True))

        # ----------------------------------------------------------- #
        # Version 5: contrastive loss
        # ----------------------------------------------------------- #

        self.SELayer = SELayer(512, self.top_k)
        # self.sp_fc = nn.Sequential(nn.Dropout(0.2),
        #                             nn.Linear(self.top_k * 11 * 11, sp_embedding_feature_dim))
        self.sp_fc = nn.Sequential(nn.Linear(self.top_k * 11 * 11, sp_embedding_feature_dim))

        self.head = nn.Sequential(
                nn.Linear(sp_embedding_feature_dim, sp_embedding_feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(sp_embedding_feature_dim, 128)
            )
                                    

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x) # (batch_size, 64, 84, 84)
        x2 = self.layer2(x1) # (batch_size, 128, 42, 42)
        x3 = self.layer3(x2) # (batch_size, 256, 21, 21)
        x4 = self.layer4(x3) # (batch_size, 512, 11, 11)

        x5 = self.avgpool(x4)
        feature = x5.view(x5.size(0), -1)


        # ----------------------------------------------------------- #
        # Version 3
        # ----------------------------------------------------------- #
        # att_feature = self.SELayer(x4)
        # att_feature = att_feature.view(x.size(0), -1)
        # att_feature = self.sp_fc(att_feature)

        # ----------------------------------------------------------- #
        # Version 4
        # ----------------------------------------------------------- #
        # att_feature = self.sp_layer(x3)
        # att_feature = self.avgpool(att_feature)
        # att_feature = att_feature.view(x.size(0), -1)


        # import ipdb; ipdb.set_trace()
        # att_feature = self.pool(self.SELayer3(x3)[0])
        # att_feature = att_feature.view(x.size(0), -1)


        # ----------------------------------------------------------- #
        # Version 5: contrastive loss on sp_feature
        # ----------------------------------------------------------- #
        att_feature = self.SELayer(x4)
        att_feature = att_feature.view(x.size(0), -1)
        att_feature = self.sp_fc(att_feature)
        contrastive_feature = F.normalize(self.head(att_feature), dim=1)

        return feature, self.fc(feature), att_feature, contrastive_feature


def resnet10_att(num_classes, sp_embedding_feature_dim, pool_type, top_k):
    """Constructs a ResNet-10-attention model.
    """
    model = ResNetAttention(BasicBlock, [1, 1, 1, 1], num_classes, sp_embedding_feature_dim, pool_type, top_k)
    return model


def resnet18_att(num_classes, sp_embedding_feature_dim, pool_type, top_k):
    """Constructs a ResNet-18-attention model.
    """
    model = ResNetAttention(BasicBlock, [2, 2, 2, 2], num_classes, sp_embedding_feature_dim, pool_type, top_k)
    return model


def resnet34_att(num_classes, sp_embedding_feature_dim, pool_type, top_k):
    """Constructs a ResNet-34-att model.
    """
    model = ResNetAttention(BasicBlock, [3, 4, 6, 3], num_classes, sp_embedding_feature_dim, pool_type, top_k)
    return model


def resnet50_att(num_classes, sp_embedding_feature_dim, pool_type, top_k):
    """Constructs a ResNet-50-att model.
    """
    model = ResNetAttention(Bottleneck, [3, 4, 6, 3], num_classes, sp_embedding_feature_dim)
    return model


def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
