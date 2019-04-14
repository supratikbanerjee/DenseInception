from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InceptionV4(nn.Module):

    def __init__(self, growth_rate, nblocks, reduction):
        super(InceptionV4, self).__init__()
        self.A_size = 2*growth_rate
        self.T1i_size = self.A_size + 128 * nblocks[0]
        self.T1o_size = int(math.floor((2*growth_rate)+(nblocks[0]*growth_rate)*reduction))
        self.B_size = self.T1o_size
        self.T2i_size = self.B_size + 384 * nblocks[1]
        self.T2o_size = int(math.floor((2*growth_rate)+(nblocks[1]*growth_rate)*reduction))
        self.C_size = self.T2o_size
        self.L_size = self.C_size + 448 * nblocks[2]
        self.features = nn.Sequential(
            DenseInceptionStem(growth_rate),
            InceptionA(self.A_size),
            InceptionA(self.A_size + 128),
            InceptionA(self.A_size + 128 * 2),
            InceptionA(self.A_size + 128 * 3),
            InceptionA(self.A_size + 128 * 4),
            #ReductionA(),
            Transition(self.T1i_size, self.T1o_size),
            InceptionB(self.B_size),
            InceptionB(self.B_size + 384),
            InceptionB(self.B_size + 384 * 2),
            InceptionB(self.B_size + 384 * 3),
            InceptionB(self.B_size + 384 * 4),
            InceptionB(self.B_size + 384 * 5),
            InceptionB(self.B_size + 384 * 6),
            InceptionB(self.B_size + 384 * 7),
            InceptionB(self.B_size + 384 * 8),
            InceptionB(self.B_size + 384 * 9),
            #ReductionB(),
            Transition(self.T2i_size, self.T2o_size),
            InceptionC(self.C_size),
            InceptionC(self.C_size + 448),
            InceptionC(self.C_size + 448 * 2),
            InceptionC(self.C_size + 448 * 3),
            InceptionC(self.C_size + 448 * 4)
            #Transition(3072, 1536)
        )

        self.AvgPool_3x3 = nn.AvgPool2d(kernel_size=7, count_include_pad=False)

        self.fc1 = nn.Linear(in_features=self.L_size, out_features=10)

        nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, x):
        x = self.features(x)
        #print(x.size())
        x = self.AvgPool_3x3(x)
        #print(x.size())
        x = nn.Dropout(p=0.5)(x)
        x = x.view(-1, self.L_size)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class InceptionC(nn.Module):
    def __init__(self, input_size):
        super(InceptionC, self).__init__()
        # Branch 1C
        self.Conv2d_iC_B11_1x1 = iconv2D(input_size, 192, kernel_size=1, stride=1)

        # Branch 2C
        self.Conv2d_iC_B2_1x1 = iconv2D(input_size, 192, kernel_size=1, stride=1)
        self.Conv2d_iC_B2a_1x3 = iconv2D(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.Conv2d_iC_B2b_3x1 = iconv2D(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))


    def forward(self, x):
        residual = x
        # Branch 1C
        x1 = self.Conv2d_iC_B11_1x1(x)

        # Branch 2C
        x2 = self.Conv2d_iC_B2_1x1(x)
        x2 = self.Conv2d_iC_B2a_1x3(x2)
        x2 = self.Conv2d_iC_B2b_3x1(x2)

        # Filter Concat 1C
        x = torch.cat([x1, x2, residual], 1)
        #print("C")
        #print(x.size())

        return x


class InceptionB(nn.Module):
    def __init__(self, input_size):
        super(InceptionB, self).__init__()
        # Branch 1B
        self.Conv2d_iB_B11_1x1 = iconv2D(input_size, 192, kernel_size=1)

        # Branch 2B
        self.Conv2d_iB_B21_1x1 = iconv2D(input_size, 128, kernel_size=1)
        self.Conv2d_iB_B22_1x7 = iconv2D(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.Conv2d_iB_B23_7x1 = iconv2D(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))

    def forward(self, x):
        residual = x
        # Branch 1B
        x1 = self.Conv2d_iB_B11_1x1(x)

        # Branch 2B
        x2 = self.Conv2d_iB_B21_1x1(x)
        x2 = self.Conv2d_iB_B22_1x7(x2)
        x2 = self.Conv2d_iB_B23_7x1(x2)

        # Filter Concat 1B
        x = torch.cat([x1, x2, residual], 1)
        #print("B")
        #print(x.size())

        return x


class InceptionA(nn.Module):
    def __init__(self, input_size):
        super(InceptionA, self).__init__()
        # Branch 1A
        self.Conv2d_iA_B11_1x1 = iconv2D(input_size, 32, kernel_size=1)

        # Branch 2A
        self.Conv2d_iA_B21_1x1 = iconv2D(input_size, 32, kernel_size=1)
        self.Conv2d_iA_B22_3x3 = iconv2D(32, 32, kernel_size=3, padding=1)

        # Branch 3A
        self.Conv2d_iA_B31_1x1 = iconv2D(input_size, 32, kernel_size=1)
        self.Conv2d_iA_B32_3x3 = iconv2D(32, 48, kernel_size=3, padding=1)
        self.Conv2d_iA_B33_3x3 = iconv2D(48, 64, kernel_size=3, padding=1)


    def forward(self, x):
        residual = x
        # Branch 1A
        x1 = self.Conv2d_iA_B11_1x1(x)
        #print('x1',x1.size())

        # Branch 2A
        x2 = self.Conv2d_iA_B21_1x1(x)
        x2 = self.Conv2d_iA_B22_3x3(x2)
        #print('x2',x2.size())

        # Branch 3A
        x3 = self.Conv2d_iA_B31_1x1(x)
        x3 = self.Conv2d_iA_B32_3x3(x3)
        x3 = self.Conv2d_iA_B33_3x3(x3)
        #print('x3',x3.size())

        # Filter Concat 1A
        #print(residual.size())
        x = torch.cat([x1, x2, x3, residual], 1)
        #print("A")
        #print(x.size())

        return x


class DenseInceptionStem(nn.Module):
    def __init__(self, growth_rate):
        super(DenseInceptionStem, self).__init__()
        self.Conv2d_3x3 = iconv2D(1, 2*growth_rate, kernel_size=3, padding=1)  # modified stride to adjust for 28x28 image

    def forward(self, x):
        x = self.Conv2d_3x3(x)
        return x


class iconv2D(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(iconv2D, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.bn(x)
        x =self.relu(x)
        x = self.conv(x)
        return x

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        #print(out.size())
        out = F.avg_pool2d(out, 2)
        #print(out.size())
        return out