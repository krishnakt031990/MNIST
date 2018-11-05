import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()

        self.layer1 = self._make_conv_layer(1, 32)
        self.layer2 = self._make_conv_layer(32, 64)
        self.layer3 = self._make_conv_layer(128, 512)
        self.flatten = Flatten()
        self.drop_out = nn.Dropout()
        self.fc1 = self._make_fc_layer(3 * 3 * 512, 1000)
        self.fc2 = self._make_fc_layer(1000, 500)
        self.fc3 = self._make_fc_layer(500, num_classes)

    def _make_conv_layer(self, in_channel, out_channel, conv_kernel_size=5, conv_stride=1, padding=2, mp_kernel=2,
                         mp_stride=2):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=conv_kernel_size, stride=conv_stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=mp_kernel, stride=mp_stride))
        return layer

    def _make_fc_layer(self, in_channel, out_channel):
        fc = nn.Linear(in_channel, out_channel)
        return fc

    def forward(self, x):
        out = self.layer1(x)
        out1 = self.layer2(out)
        out2 = self.layer2(out)
        out = torch.cat((out1, out2), dim=1)
        out = self.layer3(out)
        out = self.flatten(out)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
