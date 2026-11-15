from torch import nn

INPUT_SIZE = 3 * 32 * 32
NUM_CLASSES = 10

ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


def make_activation(name):
    return ACTIVATIONS[name]()


class MLP(nn.Module):
    # 3072 inputs -> hidden layers (one Linear + activation each) -> 10 logits
    def __init__(self, hidden_sizes, activation="relu"):
        super().__init__()
        layers = [nn.Flatten()]  # N x 3 x 32 x 32 -> N x 3072
        in_size = INPUT_SIZE
        for size in hidden_sizes:
            layers.append(nn.Linear(in_size, size))
            layers.append(make_activation(activation))
            in_size = size
        layers.append(nn.Linear(in_size, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    # Blocks of conv -> [BatchNorm] -> activation -> 2x2 pool, then a small fully connected head
    def __init__(self, channels, kernel_size=3, pool="max", batchnorm=False,
                 dropout=0.0, activation="relu", fc_hidden=256):
        super().__init__()
        layers = []
        in_channels, size = 3, 32
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"))
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(make_activation(activation))
            layers.append(nn.MaxPool2d(2) if pool == "max" else nn.AvgPool2d(2))
            in_channels, size = out_channels, size // 2  # pooling halves height and width

        layers.append(nn.Flatten())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_channels * size * size, fc_hidden))
        layers.append(make_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(fc_hidden, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
