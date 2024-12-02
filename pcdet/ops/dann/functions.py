from torch.autograd import Function
import torch.nn as nn


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # output = grad_output * ctx.alpha
        output = grad_output.neg() * ctx.alpha

        return output, None


# From https://github.com/nagisa-koyama/DA_detection/blob/master/lib/nets/discriminator_img.py
class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels=1, inter_channels=64):
        super(DomainDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(inter_channels, out_channels, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x


# From https://github.com/nagisa-koyama/Domain-Adaptive-Faster-RCNN-PyTorch/blob/master/maskrcnn_benchmark/modeling/da_heads/da_heads.py
class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()
        self.mid_channels = 64
        self.conv1_da = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(self.mid_channels, 1, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

        for l in [self.conv1_da, self.conv2_da]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # img_features = []
        # for feature in x:
        #     t = self.relu(self.conv1_da(feature))
        #     img_features.append(self.conv2_da(t))
        x = self.conv1_da(x)
        x = self.relu(x)
        x = self.conv2_da(x)
        return x
