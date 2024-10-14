from torch.autograd import Function
import torch.nn as nn


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


# Refers to file:///Users/nagisa/Downloads/1910.11319v1.pdf
class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels = 1, inter_channels=64):
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
