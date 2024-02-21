import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU cell
    """

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int):
        super().__init__()

        # 计算填充大小
        padding = kernel_size // 2

        self.input_size = input_size
        self.hidden_size = hidden_size

        # 重置门的卷积层、更新门的卷积层和输出门的卷积层
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        # 初始化权重和偏置
        for layer in [self.reset_gate, self.update_gate, self.out_gate]:
            init.orthogonal_(layer.weight)
            init.constant_(layer.bias, 0.)

    def forward(self, input_: torch.Tensor, prev_state: torch.Tensor = None) -> torch.Tensor:
        # 获取批次、通道、高度和宽度维度
        batch_size, _, height, width = input_.shape

        # 如果没有提供 prev_state，则生成一个全零 prev_state 张量
        if prev_state is None:
            state_size = [batch_size, self.hidden_size, height, width]
            prev_state = torch.zeros(state_size, device=input_.device)

        # 沿着通道维度拼接输入和 prev_state
        stacked_inputs = torch.cat([input_, prev_state], dim=1)

        # 计算重置门、更新门和输出门
        reset_gate_output = self.reset_gate(stacked_inputs)
        reset = torch.sigmoid(reset_gate_output)

        update_gate_output = self.update_gate(stacked_inputs)
        update = torch.sigmoid(update_gate_output)

        out_gate_output = self.out_gate(torch.cat([input_, prev_state * reset], dim=1))
        out_inputs = torch.tanh(out_gate_output)

        # 计算新状态
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state

class ConvGRUModule(nn.Module):
    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
        super(ConvGRUModule, self).__init__()

        self.input_size = input_size

        # 设置隐藏层大小
        if not isinstance(hidden_sizes, list):
            self.hidden_sizes = [hidden_sizes] * n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes`必须与n_layers具有相同的长度'
            self.hidden_sizes = hidden_sizes

        # 设置卷积核大小
        if not isinstance(kernel_sizes, list):
            self.kernel_sizes = [kernel_sizes] * n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes`必须与n_layers具有相同的长度'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        # 创建多层GRU单元
        cells = nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            # 创建单个GRU单元并添加到单元列表中
            cell = ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def forward(self, x, hidden=None):
        input_ = x
        cell_hidden = hidden
        upd_hidden = []

        # 逐层前向计算
        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]

            # 通过该层
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)

            # 更新输入到最后一层更新的隐藏层，用于下一次传递
            input_ = upd_cell_hidden
            cell_hidden = upd_cell_hidden

        # 保留张量列表以允许不同的隐藏层大小
        return upd_hidden