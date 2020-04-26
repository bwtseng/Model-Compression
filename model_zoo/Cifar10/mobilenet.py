import torch
import torch.nn as nn
import sys
import os
from math import floor
# Otherwise, keep uisng distiler's module is fine.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Access root folder
sys.path.append(BASE_DIR)
from modules import *
__all__ = ['MobileNetV2', 'mobilenet_v2', 'MobileNetV1', 'create_mobilenet_cifar10']

class MobileNetV1(nn.Module):
    def __init__(self, channel_multiplier=1.0, min_channels=8, num_classes=10):
        super(MobileNetV1, self).__init__()

        if channel_multiplier <= 0:
            raise ValueError('channel_multiplier must be >= 0')

        def conv_bn_relu(n_ifm, n_ofm, kernel_size, stride=1, padding=0, groups=1):
            return [
                nn.Conv2d(n_ifm, n_ofm, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(n_ofm),
                nn.ReLU(inplace=True)
            ]

        def depthwise_conv(n_ifm, n_ofm, stride):
            return nn.Sequential(
                *conv_bn_relu(n_ifm, n_ifm, 3, stride=stride, padding=0, groups=n_ifm),
                *conv_bn_relu(n_ifm, n_ofm, 1, stride=1)
            )

        base_channels = [32, 64, 128, 256, 512, 1024]
        self.channels = [max(floor(n * channel_multiplier), min_channels) for n in base_channels]
        #print(self.channels)
        self.model = nn.Sequential(
            nn.Sequential(
            *conv_bn_relu(3, self.channels[0], 3, stride=1, padding=1)),
            depthwise_conv(self.channels[0], self.channels[1], 1),
            depthwise_conv(self.channels[1], self.channels[2], 1), #2
            depthwise_conv(self.channels[2], self.channels[2], 1),
            depthwise_conv(self.channels[2], self.channels[3], 1), #2
            depthwise_conv(self.channels[3], self.channels[3], 1),
            depthwise_conv(self.channels[3], self.channels[4], 1), #2
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[5], 1), #2
            depthwise_conv(self.channels[5], self.channels[5], 1),
            nn.AvgPool2d(4),
        )
        #self.fc = nn.Linear(self.channels[5], num_classes)
        self.fc = nn.Linear(1024, num_classes)
    def forward(self, x):
        x = self.model(x)
        #print(x.shape)
        #x = x.view(-1, x.size(1))
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.add = EltwiseAdd()
    def forward(self, x):
        if self.use_res_connect:
            #return x + self.conv(x)
            return self.add(x, self.conv(x))
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        
        ## CIFAR10
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1], # Stride 2 -> 1 for CIFAR-10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        ## END

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        
        # CIFAR10: stride 2 -> 1
        features = [ConvBNReLU(3, input_channel, stride=1)]
        # END
        
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        self.mean = Mean([2,3])

    def forward(self, x):
        x = self.features(x)
        #x = x.mean([2, 3])
        x = self.mean(x)
        x = self.classifier(x)
        return x

def mobilenet_v1(pretrained, progress=True, width_mult=1.0, device=None, **kwargs):
    model = MobileNetV1(channel_multiplier=width_mult)
    
    if pretrained:
        """
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v1'],
                                              progress=progress)
        if device == 'cpu':
            # Remove the module appeared in the name of whole structure.
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            print("Remove module string in loaded model !!!")
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        """
        print("You need train this model first.")
    return model


def mobilenet_v2(pretrained=False, progress=True, width_mult=1.0, device='cpu', **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(num_classes=10, width_mult=width_mult, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(script_dir+'/state_dicts/mobilenet_v2.pt', map_location=device)
        model.load_state_dict(state_dict)
    return model

def create_mobilenet_cifar10(arch, pretrained, width_mult=1.0, device=None):
    if arch == 'mobilenet_v1':
        model = mobilenet_v1(pretrained=pretrained, channel_multiplier=width_mult, device=device)
    elif arch =='mobilenet_v2':
        model = mobilenet_v2(pretrained=pretrained, width_mult=width_mult, device=device)
    else:
        raise ValueError('Not support this kind of mobilenet model !!!')
    return model

# TODO add mobilenet v1 here


if __name__ == '__main__':
    model = create_mobilenet('mobilenetv1', False)
    print(model)
