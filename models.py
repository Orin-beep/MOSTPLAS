# coding = utf-8
import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self,
                 feature_dim=256,
                 class_num=6):
        super(Network, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, self.class_num),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc(x)
        out = self.classifier(x)
        
        return x, out
        

class ConvNet(nn.Module):
    def __init__(self,
                 length=200,
                 feature_dim=256,
                 kernel_num=512,
                 class_num=6):
        super(ConvNet, self).__init__()
        self.length = length
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.kernel_num = kernel_num

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.kernel_num, kernel_size=(2, self.feature_dim), stride=1, bias=False),
            nn.BatchNorm2d(self.kernel_num),
            nn.ReLU(),
        )
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.kernel_num, kernel_size=(3, self.feature_dim), stride=1, bias=False),
            nn.BatchNorm2d(self.kernel_num),
            nn.ReLU(),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.kernel_num, kernel_size=(4, self.feature_dim), stride=1, bias=False),
            nn.BatchNorm2d(self.kernel_num),
            nn.ReLU(),
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.kernel_num, kernel_size=(5, self.feature_dim), stride=1, bias=False),
            nn.BatchNorm2d(self.kernel_num),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveMaxPool2d((1,1))
        
        self.out = nn.Sequential(
            nn.Linear(4 * self.kernel_num, self.kernel_num),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.kernel_num, self.class_num),
            nn.Sigmoid(),
        )
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')   
        

    def forward(self, x):
        x_1 = self.conv_1(x)
        x_1 = self.pool(x_1)

        x_2 = self.conv_2(x)
        x_2 = self.pool(x_2)

        x_3 = self.conv_3(x)
        x_3 = self.pool(x_3)

        x_4 = self.conv_4(x)
        x_4 = self.pool(x_4)
        
        x_out = torch.cat([x_1, x_2, x_3, x_4],dim=1)
        x_out = torch.flatten(x_out,1)
        x_out = self.out(x_out)
        output = self.classifier(x_out)

        return x_out, output
        
        
class Multi_conv_Network(nn.Module):
    def __init__(self,
                 length=200,
                 feature_dim=256,
                 kernel_num=256,
                 class_num=6):
        super(Multi_conv_Network, self).__init__()
        
        self.length = length
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.kernel_num = kernel_num
        
        self.fc = Network(feature_dim=self.feature_dim, class_num=self.class_num)
        self.conv = ConvNet(feature_dim=self.feature_dim, kernel_num=self.kernel_num, class_num=self.class_num)
    
    def forward(self, x, x_sentence):
        x, out_1 = self.fc(x)
        x_sentence, out_2 = self.conv(x_sentence)

        return x, out_1, x_sentence, out_2