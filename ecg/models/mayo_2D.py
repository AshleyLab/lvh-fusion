import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class MayoV1(nn.Module):
    '''Mayo model from Screening for cardiac contractile dysfunction
        using an artificial intelligence–enabled
        electrocardiogram
        https://www-nature-com.stanford.idm.oclc.org/articles/s41591-018-0240-2.pdf
        '''
    def __init__(self, num_classes=2, input_shape=(1, 12, 5000)):
        super(MayoV1, self).__init__()
        # 1 input image channel, 16 output channels, kx1 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 5))
        self.bn1 = nn.BatchNorm2d(16)
        self.mp1 =nn.MaxPool2d(kernel_size=(1,2))

        self.conv2 = nn.Conv2d(16, 16, kernel_size=(1, 5))
        self.bn2 = nn.BatchNorm2d(16)
        self.mp2 =nn.MaxPool2d(kernel_size=(1,2))

        self.conv3 = nn.Conv2d(16, 32, kernel_size=(1, 5))
        self.bn3 = nn.BatchNorm2d(32)
        self.mp3 =nn.MaxPool2d(kernel_size=(1,4))

        self.conv4 = nn.Conv2d(32, 32, kernel_size=(1, 3))
        self.bn4 = nn.BatchNorm2d(32)
        self.mp4 =nn.MaxPool2d(kernel_size=(1,2))

        self.conv5 = nn.Conv2d(32, 64, kernel_size=(1, 3))
        self.bn5 = nn.BatchNorm2d(64)
        self.mp5 =nn.MaxPool2d(kernel_size=(1,2))

        self.conv6 = nn.Conv2d(64, 64, kernel_size=(1, 3))
        self.bn6 = nn.BatchNorm2d(64)
        self.mp6 =nn.MaxPool2d(kernel_size=(1,4))

        self.conv7 = nn.Conv2d(64, 64, kernel_size=(12, 1))
        self.bn7 = nn.BatchNorm2d(64)
        
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64*1*18, 64)  # 6*6 from image dimension
        self.bn8 = nn.BatchNorm1d(64)
        self.do1 = nn.Dropout(p=0.1)

        self.fc2 = nn.Linear(64, 32)  # 6*6 from image dimension
        self.bn9 = nn.BatchNorm1d(32)
        self.do2 = nn.Dropout(p=0.1)
     
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        # Max pooling over a (1, 2) window
        x = self.mp1(self.bn1(F.relu(self.conv1(x))))
        x = self.mp2(self.bn2(F.relu(self.conv2(x))))
        x = self.mp3(self.bn3(F.relu(self.conv3(x))))

        x = self.mp4(self.bn4(F.relu(self.conv4(x))))
        x = self.mp5(self.bn5(F.relu(self.conv5(x))))
        x = self.mp6(self.bn6(F.relu(self.conv6(x))))
        x = self.bn7(F.relu(self.conv7(x)))

        #print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        x = self.bn8(F.relu(self.fc1(x)))
        x = self.do1(x)
        x = self.bn9(F.relu(self.fc2(x)))
        x = self.do2(x)
        x = self.classifier(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class MayoV2(nn.Module):
    '''Detection of Hypertrophic Cardiomyopathy Using a Convolutional Neural Network-Enabled Electrocardiogram'''
    def __init__(self, num_classes=2, input_shape=(1, 12, 5000)):
        super(MayoV2, self).__init__()
        # 1 input image channel, 16 output channels, 12xk square convolution
        # kernel
        self.conv1a = nn.Conv2d(1, 16, kernel_size=(5, 1), padding=(5,1))
        self.bn1a = nn.BatchNorm2d(16)
        self.conv1b = nn.Conv2d(16, 16, kernel_size=(5, 1), padding=(5,1))
        self.bn1b = nn.BatchNorm2d(16)
        self.mp1 = nn.MaxPool2d(kernel_size=(1, 2))


        self.conv2a = nn.Conv2d(16, 16, kernel_size=(5, 1), padding=(5,1))
        self.bn2a = nn.BatchNorm2d(16)
        self.conv2b = nn.Conv2d(16, 16, kernel_size=(5, 1), padding=(5,1))
        self.bn2b = nn.BatchNorm2d(16)
        self.mp2 =nn.MaxPool2d(kernel_size=(1, 2))

        
        self.conv3a = nn.Conv2d(16, 32, kernel_size=(5, 1), padding=(5,1))
        self.bn3a = nn.BatchNorm2d(32)
        self.conv3b = nn.Conv2d(32, 32, kernel_size=(5, 1), padding=(5,1))
        self.bn3b = nn.BatchNorm2d(32)
        self.mp3 =nn.MaxPool2d(kernel_size=(1, 4))


        self.conv4a = nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(3,1))
        self.bn4a = nn.BatchNorm2d(32)
        self.conv4b = nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(3,1))
        self.bn4b = nn.BatchNorm2d(32)
        self.mp4 = nn.MaxPool2d(kernel_size=(1, 2))


        self.conv5a = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(3,1))
        self.bn5a = nn.BatchNorm2d(64)
        self.conv5b = nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(3,1))
        self.bn5b = nn.BatchNorm2d(64)
        self.mp5 =nn.MaxPool2d(kernel_size=(1, 2))


        self.conv6a = nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(3,1))
        self.bn6a = nn.BatchNorm2d(64)
        self.conv6b = nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(3,1))
        self.bn6b = nn.BatchNorm2d(64)
        self.mp6 =nn.MaxPool2d(kernel_size=(1, 2))

        self.aap = nn.AdaptiveAvgPool2d(output_size=(12,1))
        
        self.classifier = nn.Linear(64*12, num_classes)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.bn1a(F.relu(self.conv1a(x)))
        x = self.bn1b(F.relu(self.conv1b(x)))
        x = self.mp1(x)
        
        x = self.bn2a(F.relu(self.conv2a(x)))
        x = self.bn2b(F.relu(self.conv2b(x)))
        x = self.mp2(x)
        
        x = self.bn3a(F.relu(self.conv3a(x)))
        x = self.bn3b(F.relu(self.conv3b(x)))
        x = self.mp3(x)
        
        x = self.bn4a(F.relu(self.conv4a(x)))
        x = self.bn4b(F.relu(self.conv4b(x)))
        x = self.mp4(x)
        
        x = self.bn5a(F.relu(self.conv5a(x)))
        x = self.bn5b(F.relu(self.conv5b(x)))
        x = self.mp5(x)
        
        x = self.bn6a(F.relu(self.conv6a(x)))
        x = self.bn6b(F.relu(self.conv6b(x)))
        x = self.mp6(x)
        #print(x.size())
        x = self.aap(x)
        
        #print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        x = self.classifier(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class MayoV3(nn.Module):
    '''Detection of Hypertrophic Cardiomyopathy Using a Convolutional Neural Network-Enabled Electrocardiogram'''
    def __init__(self, num_classes=2, input_shape=(1, 12, 5000)):
        super(MayoV3, self).__init__()
        # 1 input image channel, 16 output channels, 12xk square convolution
        # kernel
        self.conv1a = nn.Conv2d(1, 16, kernel_size=(12, 5), padding=(12, 5))
        self.bn1a = nn.BatchNorm2d(16)
        self.conv1b = nn.Conv2d(16, 16, kernel_size=(12, 5), padding=(12, 5))
        self.bn1b = nn.BatchNorm2d(16)
        self.mp1 = nn.MaxPool2d(kernel_size=(1, 2))


        self.conv2a = nn.Conv2d(16, 16, kernel_size=(12, 5), padding=(12, 5))
        self.bn2a = nn.BatchNorm2d(16)
        self.conv2b = nn.Conv2d(16, 16, kernel_size=(12, 5), padding=(12, 5))
        self.bn2b = nn.BatchNorm2d(16)
        self.mp2 =nn.MaxPool2d(kernel_size=(1, 2))

        
        self.conv3a = nn.Conv2d(16, 32, kernel_size=(12, 5), padding=(12, 5))
        self.bn3a = nn.BatchNorm2d(32)
        self.conv3b = nn.Conv2d(32, 32, kernel_size=(12, 5), padding=(12, 5))
        self.bn3b = nn.BatchNorm2d(32)
        self.mp3 =nn.MaxPool2d(kernel_size=(1, 4))


        self.conv4a = nn.Conv2d(32, 32, kernel_size=(12, 3), padding=(12, 3))
        self.bn4a = nn.BatchNorm2d(32)
        self.conv4b = nn.Conv2d(32, 32, kernel_size=(12, 3), padding=(12, 3))
        self.bn4b = nn.BatchNorm2d(32)
        self.mp4 = nn.MaxPool2d(kernel_size=(1, 2))


        self.conv5a = nn.Conv2d(32, 64, kernel_size=(12, 3), padding=(12, 3))
        self.bn5a = nn.BatchNorm2d(64)
        self.conv5b = nn.Conv2d(64, 64, kernel_size=(12, 3), padding=(12, 3))
        self.bn5b = nn.BatchNorm2d(64)
        self.mp5 =nn.MaxPool2d(kernel_size=(1, 2))


        self.conv6a = nn.Conv2d(64, 64, kernel_size=(12, 3), padding=(12, 3))
        self.bn6a = nn.BatchNorm2d(64)
        self.conv6b = nn.Conv2d(64, 64, kernel_size=(12, 3), padding=(12, 3))
        self.bn6b = nn.BatchNorm2d(64)
        self.mp6 =nn.MaxPool2d(kernel_size=(1, 2))

        self.aap = nn.AdaptiveAvgPool2d(output_size=(12,1))
        
        self.classifier = nn.Linear(64*12, num_classes)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.bn1a(F.relu(self.conv1a(x)))
        x = self.bn1b(F.relu(self.conv1b(x)))
        x = self.mp1(x)
        
        x = self.bn2a(F.relu(self.conv2a(x)))
        x = self.bn2b(F.relu(self.conv2b(x)))
        x = self.mp2(x)
        
        x = self.bn3a(F.relu(self.conv3a(x)))
        x = self.bn3b(F.relu(self.conv3b(x)))
        x = self.mp3(x)
        
        x = self.bn4a(F.relu(self.conv4a(x)))
        x = self.bn4b(F.relu(self.conv4b(x)))
        x = self.mp4(x)
        
        x = self.bn5a(F.relu(self.conv5a(x)))
        x = self.bn5b(F.relu(self.conv5b(x)))
        x = self.mp5(x)
        
        x = self.bn6a(F.relu(self.conv6a(x)))
        x = self.bn6b(F.relu(self.conv6b(x)))
        x = self.mp6(x)
        #print(x.size())
        x = self.aap(x)
        
        #print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        x = self.classifier(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




def test_MayoV1():
    net = MayoV1()
    print(net)
    y = net(torch.randn(1, 1, 12, 5000))
    print(y.size())



def test_MayoV2():
    net = MayoV2()
    print(net)
    y = net(torch.randn(1, 1, 12, 5000))
    print(y.size())



def test_MayoV3():
    net = MayoV3()
    print(net)
    y = net(torch.randn(1, 1, 12, 5000))
    print(y.size())