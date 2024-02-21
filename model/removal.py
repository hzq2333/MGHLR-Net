import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from model.efficient import get_efficientnet
from model.unet_optimization import UNetModule
from model.convgru1 import ConvGRUCell

model_infos = {
    'b5':[24,40,64,176,2048], 'b4':[24,32,56,160,1792], 'b3':[24,32,48,136,1536], 
    'b2':[16,24,48,120,1408], 'b1':[16,24,40,112,1280], 'b0':[16,24,40,112,1280],
}



class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(inplace=True))

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)

        

class EfficientEncoder(nn.Module):
    def __init__(self, backbone):
        super(EfficientEncoder, self).__init__()
        self.original_model = get_efficientnet(backbone)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features

class Decoder(nn.Module):
    def __init__(self, n_channel_features):
        super(Decoder, self).__init__()
        features = n_channel_features[-1]

        self.conv2 = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=features // 1 + n_channel_features[-2], output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + n_channel_features[-3], output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + n_channel_features[-4], output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + n_channel_features[-5], output_features=features // 16)

        self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        out = self.conv3(x_d4)
        return torch.tanh(out)

class lightnessAdjustModule(nn.Module):
    def __init__(self, n_class, ch_in, ch_base, ch_hidden, n_loop=2):
        super(lightnessAdjustModule, self).__init__()
        self.n_loop = n_loop
        # features extractor
        self.encoder = UNetModule(n_class, ch_in, ch_base, False)
        # convgru
        self.gru_cell = ConvGRUCell(ch_base, hidden_size=ch_hidden, kernel_size=3)

        # output
        self.output = nn.Sequential(
            nn.Conv2d(ch_hidden, n_class, 1),
            nn.Tanh()
        )

    def forward(self, x, hidden=None):
        fea = self.encoder(x)
        for _ in range(self.n_loop):
            hidden = self.gru_cell(fea, hidden)

        output = self.output(hidden) # scale factor for v (hsv)
        return output, hidden


class highlightRemoval(nn.Module):
    def __init__(self, n_cells=4):
        super(highlightRemoval, self).__init__()

        # 记录网络中的cell数量
        self.n_layers = n_cells

        # 创建一组 lightnessAdjustModule 作为网络的编码器部分
        self.convgru_cells = nn.ModuleList([lightnessAdjustModule(3, 4) for _ in range(n_cells)])

    def forward(self, rgb, mask):
        # 用于存储不同层的预测输出
        pred_rgbs = []

        # 初始化隐藏状态为 None
        hidden_state = None

        # 初始化当前预测输出为原始 RGB 图像
        cur_pred = rgb

        # 逐层计算预测输出
        for i in range(self.n_layers):
            # 将当前预测输出和mask图像在通道维度上拼接
            input = torch.cat((cur_pred, mask), axis=1)

            # 获取当前层的 lightnessAdjustModule
            convgru_cell = self.convgru_cells[i] 

            # 使用当前层的 lightnessAdjustModule 计算预测输出和新的隐藏状态
            output, new_hidden_state = convgru_cell(input, hidden_state)

            # 将隐藏状态更新为新的隐藏状态
            hidden_state = new_hidden_state

            # 将当前预测输出加上遮罩图像和调整后的输出的乘积
            cur_pred = cur_pred + mask * output

            # 将当前层的预测输出添加到结果列表中
            pred_rgbs.append(cur_pred)

        # 返回不同层的预测输出
        return pred_rgbs

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
        




class HLRNet(nn.Module):
    def __init__(self, args):
        super(HLRNet, self).__init__()
        self.backbone = args.backbone
        # 编码器
        n_channel_features = model_infos[self.backbone]
        self.efficient_encoder = EfficientEncoder(self.backbone)

        # 解码器
        features = n_channel_features[-1]

        self.conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=features // 1 + n_channel_features[-2], output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + n_channel_features[-3], output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + n_channel_features[-4], output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + n_channel_features[-5], output_features=features // 16)

        n_dim_align, ch_base = 32, 64
        self.align1 = nn.Sequential(
            nn.Conv2d(features // 2, n_dim_align, 1, bias=False), nn.BatchNorm2d(n_dim_align), nn.LeakyReLU(inplace=True)
        )
        self.align2 = nn.Sequential(
            nn.Conv2d(features // 4, n_dim_align, 1, bias=False), nn.BatchNorm2d(n_dim_align), nn.LeakyReLU(inplace=True)
        )
        self.align3 = nn.Sequential(
            nn.Conv2d(features // 8, n_dim_align, 1, bias=False), nn.BatchNorm2d(n_dim_align), nn.LeakyReLU(inplace=True)
        )
        self.align4 = nn.Sequential(
            nn.Conv2d(features // 16, n_dim_align, 1, bias=False), nn.BatchNorm2d(n_dim_align), nn.LeakyReLU(inplace=True)
        )

        self.pred_rgb = lightnessAdjustModule(3, n_dim_align + 1, ch_base, n_dim_align)
        # self.predict_mask = nn.Sequential(
        #     nn.Conv2d(n_dim_align, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(8, 1, 1), nn.Sigmoid()
        # )
        self.predict_mask = nn.Sequential(
            nn.Conv2d(n_dim_align, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            ResidualBlock(16, 16),
            nn.Conv2d(16, 8, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
            ResidualBlock(8, 8),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb):
        features = self.efficient_encoder(rgb)
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        # 上采样
        x_d0 = self.conv(x_block4)
        x_d1 = self.up1(x_d0, x_block3)

        x_d2 = self.up2(x_d1, x_block2)
        x_align2 = self.align2(x_d2)
        
        pred_mask2 = self.predict_mask(x_align2)
        hidden_state = F.interpolate(x_align2, x_block2.shape[-2:], mode='bilinear', align_corners=True)
        pred_rgb2, hidden_state = self.pred_rgb(torch.cat((x_align2, pred_mask2), dim=1), hidden_state)
        
        

        x_d3 = self.up3(x_d2, x_block1)
        x_align3 = self.align3(x_d3)
        

        x_d4 = self.up4(x_d3, x_block0)
        x_align4 = self.align4(x_d4)
        pred_mask4 = self.predict_mask(x_align4)
        hidden_state = F.interpolate(hidden_state, x_block0.shape[-2:], mode='bilinear', align_corners=True)
        pred_rgb4, _ = self.pred_rgb(torch.cat((x_align4, pred_mask4), dim=1), hidden_state)

        pred_mask2 = F.interpolate(pred_mask2, rgb.shape[-2:], mode='bilinear', align_corners=True)
        pred_mask4 = F.interpolate(pred_mask4, rgb.shape[-2:], mode='bilinear', align_corners=True)
        pred_rgb2 = F.interpolate(pred_rgb2, rgb.shape[-2:], mode='bilinear', align_corners=True)
        pred_rgb4 = F.interpolate(pred_rgb4, rgb.shape[-2:], mode='bilinear', align_corners=True)


        pred_rgb2 = torch.clamp(rgb+pred_rgb2*pred_mask2, 0.0, 1.0)
        pred_rgb4 = torch.clamp(rgb+pred_rgb4*pred_mask4, 0.0, 1.0)
        
        return [pred_mask2, pred_mask4], [pred_rgb2, pred_rgb4]

    def name(self):
        return 'DennisChen2023'
        





