import torch
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn import MaxPool1d
from torch.nn import AvgPool1d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn.functional import leaky_relu
from torch.nn.functional import relu


class CNN(torch.nn.Module):
    def __init__(self, batch_size, inputs, outputs):
        super(CNN, self).__init__()
        self.batch_size = batch_size

        self.inputs = inputs
        self.outputs = outputs
        # self.batch_normalization = BatchNorm1d(inputs)
        self.input_layer = Conv1d(inputs, batch_size, 1)
        self.max_pooling_layer = MaxPool1d(1)
        self.avg_pooling_layer = AvgPool1d(1)
        self.conv_layer = Conv1d(batch_size, 256, 1)
        self.conv_layer1 = Conv1d(256, 512, 1)
        self.conv_layer2 = Conv1d(512, 1024, 1)

        self.flatten_layer = Flatten()
        self.linear_layer = Linear(1024, 128)
        self.outputs_layer = Linear(128, outputs)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input):
        input = input.reshape((self.batch_size, self.inputs, 1))
        # output = self.batch_normaliztion(input)
        output = leaky_relu(self.input_layer(input))
        output = self.max_pooling_layer(output)
        output = leaky_relu(self.conv_layer(output))
        output = self.max_pooling_layer(output)
        output = leaky_relu(self.conv_layer1(output))
        output = self.max_pooling_layer(output)
        output = leaky_relu(self.conv_layer2(output))
        output = self.max_pooling_layer(output)

        output = self.flatten_layer(output)
        output = self.linear_layer(output)
        # output = self.dropout(output)
        output = self.outputs_layer(output)
        # output = self.dropout(output)

        return output
