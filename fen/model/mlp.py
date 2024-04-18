import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_hidden: int,
        n_out: int,
        n_layers: int,
        p_dropout: float,
    ):
        super(MLP, self).__init__()
        self.activation = nn.GELU
        self.dropout = nn.Dropout

        layers = [
            nn.Linear(n_in, n_hidden),
        ]
        for _ in range(n_layers - 2):
            layers.extend([
                self.activation(),
                self.dropout(p_dropout),
                nn.Linear(n_hidden, n_hidden),
            ])
        layers.extend([
            self.activation(),
            self.dropout(p_dropout),
            nn.Linear(n_hidden, n_out),
        ])

        self.layers = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layers(x)
