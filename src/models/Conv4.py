from torch import nn
from .ChannelAtt import SELayer

__all__ = ['Conv4', 'Conv4Att']


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class Conv4(nn.Module):
    def __init__(self, num_classes, remove_linear=False):
        super(Conv4, self).__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        if remove_linear:
            self.logits = None
        else:
            self.logits = nn.Linear(1600, num_classes)

    def forward(self, x, feature=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)
        if self.logits is None:
            if feature:
                return x, None
            else:
                return x
        if feature:
            x1 = self.logits(x)
            return x, x1

        return self.logits(x)

# -------------------------------------- #
# Version 1: max/avg pool super features
# -------------------------------------- #
# class Conv4Att(nn.Module):
#     def __init__(self, num_classes, num_spclasses, pool_type='avg_pool'):
#         super(Conv4Att, self).__init__()
#         self.conv1 = conv_block(3, 64)
#         self.conv2 = conv_block(64, 64)
#         self.conv3 = conv_block(64, 64)
#         self.conv4 = conv_block(64, 64)

#         self.dim_feature = 1600

#         self.fc = nn.Linear(1600, num_classes)
#         self.sp_fc = nn.Linear(1600, num_spclasses)
#         self.SELayer = SELayer(64)

#         if pool_type == 'avg_pool':
#             self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         elif pool_type == 'max_pool':
#             self.pool = nn.AdaptiveMaxPool2d((1, 1))

#     def forward(self, x):
#         x1 = self.conv1(x)  # (batch_size, 64, 42, 42): 64 is attention dimension in attention class
#         x2 = self.conv2(x1) # (batch_size, 64, 21, 21): 64 is attention dimension in attention class
#         x3 = self.conv3(x2) # (batch_size, 64, 10, 10): 64 is attention dimension in attention class
#         x4 = self.conv4(x3) # (batch_size, 64, 5, 5): 64 is attention dimension in attention class
        
#         feature = x4.view(x.size(0), -1)
#         # att_feature, _ = self.SELayer(x4)
#         # att_feature = att_feature.view(x.size(0), -1)

#         att_feature = self.SELayer(x4)[0]
#         att_feature = att_feature.view(x.size(0), -1)

#         return feature, self.fc(feature), att_feature, self.sp_fc(att_feature)


# ---------------------------------------------------- #
# Version 2: select top-k feature maps super features
# ---------------------------------------------------- #
class Conv4Att(nn.Module):
    def __init__(self, num_classes, num_spclasses, top_k=16):
        super(Conv4Att, self).__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)

        self.top_k = 16

        self.fc = nn.Linear(1600, num_classes)
        self.sp_fc = nn.Linear(400, num_spclasses)
        self.SELayer = SELayer(64, self.top_k)

        # if pool_type == 'avg_pool':
        #     self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # elif pool_type == 'max_pool':
        #     self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        x1 = self.conv1(x)  # (batch_size, 64, 42, 42): 64 is attention dimension in attention class
        x2 = self.conv2(x1) # (batch_size, 64, 21, 21): 64 is attention dimension in attention class
        x3 = self.conv3(x2) # (batch_size, 64, 10, 10): 64 is attention dimension in attention class
        x4 = self.conv4(x3) # (batch_size, 64, 5, 5): 64 is attention dimension in attention class
        
        feature = x4.view(x.size(0), -1)
        # att_feature, _ = self.SELayer(x4)
        # att_feature = att_feature.view(x.size(0), -1)

        att_feature = self.SELayer(x4, all=False)[0]
        att_feature = att_feature.view(x.size(0), -1)

        return feature, self.fc(feature), att_feature, self.sp_fc(att_feature)


# --------------------------------------------------------------------- #
# Version 3: match image features to the super class embedding features
# --------------------------------------------------------------------- #
# class Conv4Att(nn.Module):
#     def __init__(self, num_classes, sp_embedding_feature_dim, top_k=16):
#         super(Conv4Att, self).__init__()
#         self.conv1 = conv_block(3, 64)
#         self.conv2 = conv_block(64, 64)
#         self.conv3 = conv_block(64, 64)
#         self.conv4 = conv_block(64, 64)

#         self.top_k = 16

#         self.fc = nn.Linear(1600, num_classes)
#         self.sp_fc = nn.Linear(400, sp_embedding_feature_dim)
#         self.SELayer = SELayer(64, self.top_k)

#         # if pool_type == 'avg_pool':
#         #     self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         # elif pool_type == 'max_pool':
#         #     self.pool = nn.AdaptiveMaxPool2d((1, 1))

#     def forward(self, x):
#         x1 = self.conv1(x)  # (batch_size, 64, 42, 42): 64 is attention dimension in attention class
#         x2 = self.conv2(x1) # (batch_size, 64, 21, 21): 64 is attention dimension in attention class
#         x3 = self.conv3(x2) # (batch_size, 64, 10, 10): 64 is attention dimension in attention class
#         x4 = self.conv4(x3) # (batch_size, 64, 5, 5): 64 is attention dimension in attention class
        
#         feature = x4.view(x.size(0), -1)
#         # att_feature, _ = self.SELayer(x4)
#         # att_feature = att_feature.view(x.size(0), -1)

#         att_feature = self.SELayer(x4, all=False)[0]
#         att_feature = att_feature.view(x.size(0), -1)

#         return feature, self.fc(feature), att_feature, self.sp_fc(att_feature)
