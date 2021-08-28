import torch
from torch import nn


class AE_0_layer(nn.Module):
    def __init__(self, **kwargs):
        super(AE_0_layer, self).__init__()
        self.encoder = nn.Linear(
            in_features=kwargs["input_dim"], out_features=kwargs["compress_dim"]
        )

        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()

    def encoding(self, features):
        return self.encoder(features)

    def forward(self, features):
        encoded_emb = self.encoding(features)
        encoded_emb = torch.relu(encoded_emb)
        return encoded_emb


class AE_1_layer(nn.Module):
    def __init__(self, **kwargs):
        super(AE_1_layer, self).__init__()
        self.encoder = nn.Linear(
            in_features=kwargs["input_dim"], out_features=int(kwargs["compress_dim"])
        )
        self.decoder = nn.Linear(
            in_features=int(kwargs["compress_dim"]), out_features=kwargs["input_dim"]
        )

        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()

    def encoding(self, features):
        return self.encoder(features)
    def decoding(self, features):
        return self.decoder(features)

    def forward(self, features):
        encoded_emb = self.encoding(features)
        encoded_emb = torch.relu(encoded_emb)
        decoded_emb = self.decoding(encoded_emb)
        return decoded_emb
