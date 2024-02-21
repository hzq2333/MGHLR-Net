import torch
import torch.nn as nn
import torch.nn.functional as F        
        
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, padding=1):
        super(UNetConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.cbam = CBAMBlock(out_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.cbam(x) # 加入CBAM注意力机制
        x = self.relu(x)
        return x

class UNetModule(nn.Module):
    def __init__(self, n_class, ch_in, ch_base=32, with_output=True):
        super(UNetModule, self).__init__()
        self.with_output = with_output
        chs = [ch_base, ch_base*2, ch_base*4, ch_base*8]
        # 定义encoder部分
        self.conv_down1 = UNetConvBlock(ch_in, chs[0])
        self.conv_down2 = UNetConvBlock(chs[0], chs[1])
        self.conv_down3 = UNetConvBlock(chs[1], chs[2])
        self.conv_down4 = UNetConvBlock(chs[2], chs[3])

        self.maxpool = nn.MaxPool2d(2)
        
        # 定义decoder部分
        self.conv_up3 = UNetConvBlock(chs[3] + chs[2], chs[2])
        self.conv_up2 = UNetConvBlock(chs[2] + chs[1], chs[1])
        self.conv_up1 = UNetConvBlock(chs[1] + chs[0], chs[0])
        
        if with_output:
            # 定义输出层
            self.conv_last = nn.Conv2d(chs[0], n_class, 1)

    def forward(self, x):
        # encoder
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.conv_down4(x)
        
        # decoder
        x = F.interpolate(x, conv3.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up3(x)

        x = F.interpolate(x, conv2.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up2(x)

        x = F.interpolate(x, conv1.shape[-2:], mode='bilinear', align_corners=True) 
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_up1(x)
        
        if self.with_output:
            out = self.conv_last(x)
            return out
        
        return x        