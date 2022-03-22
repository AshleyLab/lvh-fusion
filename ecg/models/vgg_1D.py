'''VGG11/13/16/19 in Pytorch. 
    Modified from https://raw.githubusercontent.com/kuangliu/pytorch-cifar/master/models/vgg.py'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': ([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', ('AP', 1)], [('F', 4096), ('D', 0.2), ('F',4069), ('D', 0.2)]),
    'VGG13': ([64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', ('AP', 1)], [('F', 4096), ('D', 0.2), ('F',4069), ('D', 0.2)]),
    'VGG16': ([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', ('AP', 1)],[('F', 4096), ('D', 0.2), ('F',4069), ('D', 0.2)]),
    'VGG19': ([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M', ('AP', 1)], [('F', 4096), ('D', 0.2), ('F',4069), ('D', 0.2)])
}


class VGG_1D(nn.Module):
    ## vgg_name is an attribute of class VGG_test
    def __init__(self, vgg_name, num_classes=2):
        ## super is need here to make sure method resoultion order is followed. ie multiple inheriance, of classes.
        super().__init__()
        self.features, in_channels = self._make_cnn_layers(cfg[vgg_name][0])
        self.fully_connected, in_channels = self._make_fully_connected_layers(cfg[vgg_name][1], in_channels)
        self.classifier = nn.Linear(in_channels, num_classes) 

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fully_connected(out)
        out = self.classifier(out)
        return out

    ## private method, would not be imported if from vgg_test import * was used.
    ### the use of self here allows the bining of the instance to the class?
    def _make_cnn_layers(self, cfg):
        ## Empty list
        layers = []
        in_channels = 12
        for x in cfg:
            if x == 'M':
                ## adding to the list
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            if (isinstance(x, int)):
                layers += [nn.Conv1d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm1d(x),
                           nn.ReLU(inplace=True)]
                ## here we are always keeping track of the inchaneels 
                in_channels = x
            if (isinstance(x, tuple)):
                if x[0] == 'AP':
                    layers += [nn.AdaptiveAvgPool1d(output_size=x[1])]
                    ## here we are always keeping track of the inchaneels 
                    in_channels = x[1] * in_channels
        ## returning a unpacked Sequential of layers 
        print(*layers)
        return nn.Sequential(*layers), in_channels
    
    def _make_fully_connected_layers(self, cfg, in_channels):
        ## Empty list
        layers = []
        for x in cfg:
            if (isinstance(x, tuple)):
                if x[0] == 'D':
                    layers += [nn.Dropout(x[1])]
                if x[0] == 'F':
                    layers += [nn.Linear(in_channels, x[1]), nn.ReLU(inplace=True)]
                    in_channels = x[1]
        print(*layers)
        return nn.Sequential(*layers), in_channels

def test():
    net = VGG_1D('VGG11')
    x = torch.randn(1, 12, 5000)
    y = net(x)
    print(y.size())

