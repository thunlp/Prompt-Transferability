import torch
from torch import nn


class AE_auto_layer(nn.Module):
    def __init__(self, **kwargs):
        super(AE_auto_layer, self).__init__()

        dims = [v for k,v in kwargs]

        for id, dim in enumerate(dims):
            self.encoders = [nn.Linear(in_features=dim, out_features=dims[id+1])]
            if len(self.encoders) >= len(dim)-1:
                break

        #self.encoder = nn.Linear(
        #    in_features=kwargs["input_dim"], out_features=kwargs["compress_dim"]
        #)

        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()
        self.activation = nn.LeakyReLU()

    #def encoding(self, features):
    #    return self.encoder(features)

    def forward(self, features):
        '''
        encoded_emb = self.encoding(features)
        encoded_emb = torch.relu(encoded_emb)
        return encoded_emb
        '''
        for encoder in self.encoders:
            features = encoder(features)
            features = self.activation(features)
        return features



class AE_0_layer_mutiple_100(nn.Module):
    def __init__(self, **kwargs):
        super(AE_0_layer_mutiple_100, self).__init__()
        self.encoder = nn.Linear(
            in_features=kwargs["dim_0"], out_features=kwargs["dim_1"]
        )

        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()
        self.activation = nn.LeakyReLU()

    def encoding(self, features):
        return self.encoder(features)

    def forward(self, features):
        encoded_emb = self.encoding(features)
        encoded_emb = self.activation(encoded_emb)
        return encoded_emb


class AE_0_layer(nn.Module):
    def __init__(self, **kwargs):
        super(AE_0_layer, self).__init__()
        self.encoder = nn.Linear(
            in_features=kwargs["dim_0"], out_features=kwargs["dim_1"]
        )

        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()
        self.activation = nn.LeakyReLU()

    def encoding(self, features):
        return self.encoder(features)

    def forward(self, features):
        encoded_emb = self.encoding(features)
        encoded_emb = self.activation(encoded_emb)
        return encoded_emb



class AE_1_layer_mutiple_100(nn.Module):
    def __init__(self, **kwargs):
        super(AE_1_layer_mutiple_100, self).__init__()
        self.encoder = nn.Linear(
            in_features=kwargs["dim_0"], out_features=int(kwargs["dim_1"])
        )
        self.decoder = nn.Linear(
            in_features=int(kwargs["dim_1"]), out_features=kwargs["dim_2"]
        )

        self.dim = int(kwargs["dim_2"]/100)

        #########################
        self.layer_norm = nn.LayerNorm(self.dim, eps=1e-05)
        #########################

        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()
        #self.activation = nn.LeakyReLU()
        self.activation = nn.Tanh()

    def encoding(self, features):
        return self.encoder(features)
    def decoding(self, features):
        return self.decoder(features)

    def forward(self, features):
        encoded_emb = self.encoding(features)
        encoded_emb = self.activation(encoded_emb)
        decoded_emb = self.decoding(encoded_emb)
        ###
        #layer_norm = nn.LayerNorm(int(decoded_emb.shape[0]),100,768)
        decoded_emb = decoded_emb.reshape(int(decoded_emb.shape[0]),100,self.dim)
        decoded_emb = self.layer_norm(decoded_emb)
        decoded_emb = decoded_emb.reshape(int(decoded_emb.shape[0]),100*self.dim)
        ###
        return decoded_emb



class AE_1_layer(nn.Module):
    def __init__(self, **kwargs):
        super(AE_1_layer, self).__init__()
        self.encoder = nn.Linear(
            in_features=kwargs["dim_0"], out_features=int(kwargs["dim_1"])
        )
        self.decoder = nn.Linear(
            in_features=int(kwargs["dim_1"]), out_features=kwargs["dim_2"]
        )

        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()
        self.activation = nn.LeakyReLU()

    def encoding(self, features):
        return self.encoder(features)
    def decoding(self, features):
        return self.decoder(features)

    def forward(self, features):
        encoded_emb = self.encoding(features)
        encoded_emb = self.activation(encoded_emb)
        decoded_emb = self.decoding(encoded_emb)
        return decoded_emb
