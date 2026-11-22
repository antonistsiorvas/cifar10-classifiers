from torch import nn

ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


class MLP(nn.Module):

    def __init__(self, hidden, activation):
        super().__init__()
        layers = [nn.Flatten()]
        in_size = 3 * 32 * 32
        for size in hidden:
            layers.append(nn.Linear(in_size, size))
            layers.append(ACTIVATIONS[activation]())
            in_size = size
        layers.append(nn.Linear(in_size, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):

    def __init__(self, channels, kernel_size, pool, batchnorm, dropout, activation):
        super().__init__()
        layers = []
        in_channels = 3
        size = 32
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"))
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(ACTIVATIONS[activation]())
            if pool == "max":
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(nn.AvgPool2d(2))
            in_channels = out_channels
            size = size // 2

        layers.append(nn.Flatten())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_channels * size * size, 256))
        layers.append(ACTIVATIONS[activation]())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(256, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
