from torch import nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, topk=16):
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
