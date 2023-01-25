import torch.nn as nn


class OrgModel(nn.Module):
    def __init__(self, input_dim=89, hidden_units=[128], out_dim=64):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_units[0])
        hidden_layers = []
        for i in range(1, len(hidden_units)):
            hidden_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_units[-1], out_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)

        return x


class TopModel(nn.Module):
    def __init__(self, input_dim=89, hidden_units=[128], num_classes=2):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_units[0])
        hidden_layers = []
        for i in range(1, len(hidden_units)):
            hidden_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_units[-1], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)

        return x


class CentralizedModel(nn.Module):
    def __init__(self, input_dim=89, hidden_units=[128], num_classes=2):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_units[0])
        hidden_layers = []
        for i in range(1, len(hidden_units)):
            hidden_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_units[-1], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)

        return x
