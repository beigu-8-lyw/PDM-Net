import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels,
                               4 * hidden_channels, 3, padding=1)
        # Positive forget bias helps retain information at initialization.
        with torch.no_grad():
            self.gates.bias[hidden_channels:2 * hidden_channels].fill_(1.0)

    def forward(self, x, state=None):
        if state is None:
            shape = (x.shape[0], self.hidden_channels, *x.shape[-2:])
            h, c = x.new_zeros(shape), x.new_zeros(shape)
        else:
            h, c = state
        i, f, o, g = self.gates(torch.cat((x, h), dim=1)).chunk(4, dim=1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h = o.sigmoid() * c.tanh()
        return h, c


class StackedConvLSTM(nn.Module):
    def __init__(self, channels, layers=2):
        super().__init__()
        if layers < 1:
            raise ValueError("ConvLSTM requires at least one layer")
        self.cells = nn.ModuleList([ConvLSTMCell(channels, channels) for _ in range(layers)])

    def forward(self, sequence):
        if sequence.ndim != 5 or sequence.shape[1] < 1:
            raise ValueError("Expected nonempty [B,T,C,H,W]")
        states = [None] * len(self.cells)
        for x in sequence.unbind(dim=1):
            for index, cell in enumerate(self.cells):
                states[index] = cell(x, states[index])
                x = states[index][0]
        return states[-1]
