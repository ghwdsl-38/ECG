import torch
import torch.nn as nn

class ECALayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, padding=(k_size - 1) // 2, groups=channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y

class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ECABasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.eca = ECALayer(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ECGModel(nn.Module):
    def __init__(self, num_classes):
        super(ECGModel, self).__init__()

        filter_sizes = [5, 9, 11]
        self.conv1 = nn.Conv1d(1, 32, kernel_size=filter_sizes[0],
                               stride=1, bias=False, padding=(filter_sizes[0] // 2))
        self.conv2 = nn.Conv1d(1, 32, kernel_size=filter_sizes[1],
                               stride=1, bias=False, padding=(filter_sizes[1] // 2))
        self.conv3 = nn.Conv1d(1, 32, kernel_size=filter_sizes[2],
                               stride=1, bias=False, padding=(filter_sizes[2] // 2))

        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.do = nn.Dropout(0.2)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 32 * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32 * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(32 * 2, 128, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.inplanes = 128
        self.eca = self._make_layer(ECABasicBlock, 128, 1)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=25, nhead=5, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.ap = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(128, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_in):
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        x3 = self.conv3(x_in)

        x_concat = torch.mean(torch.stack([x1, x2, x3], 2), 2)
        x_concat = self.do(self.mp(self.relu(self.bn(x_concat))))

        x = self.conv_block2(x_concat)
        x = self.conv_block3(x)

        # x = self.eca(x)

        x1 = self.transformer_encoder(x)
        x2 = self.transformer_encoder(torch.flip(x, [2]))
        x = x1 + x2

        x = self.ap(x)
        x_flat = x.reshape(x.shape[0], -1)
        x_out = self.clf(x_flat)
        return x_out
    
# # Instantiate the model with the number of classes
# num_classes = 5  # example number of classes
# model = ECGModel(num_classes=num_classes)

# # Generate random input data with shape [2, 1, 186]
# input_data = torch.randn(2, 1, 186)

# # Forward pass
# output = model(input_data)
# print(output)
