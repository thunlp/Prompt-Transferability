import torch
import torch.nn as nn

#from tools.projector import AE_0_layer, AE_1_layer_mutiple_100, AE_1_layer

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
        #self.layer_norm = nn.LayerNorm(self.dim, eps=1e-05)
        #########################

        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()
        #self.activation = nn.LeakyReLU()
        #self.activation = nn.Tanh()
        self.activation = nn.Softmax(dim=0)
        #self.activation = nn.Softmax(dim=0)

    def encoding(self, features):
        return self.encoder(features)
    def decoding(self, features):
        return self.decoder(features)

    def forward(self, features):
        encoded_emb = self.encoding(features)
        encoded_emb = self.activation(encoded_emb)
        decoded_emb = self.decoding(encoded_emb)
        ###
        '''
        #layer_norm = nn.LayerNorm(int(decoded_emb.shape[0]),100,768)
        decoded_emb = decoded_emb.reshape(int(decoded_emb.shape[0]),100,self.dim)
        decoded_emb = self.layer_norm(decoded_emb)
        decoded_emb = decoded_emb.reshape(int(decoded_emb.shape[0]),100*self.dim)
        '''
        ###
        return decoded_emb


model_AE = AE_1_layer_mutiple_100(dim_0=409600,dim_1=512,dim_2=76800)
#model_AE = AE_1_layer_mutiple_100(dim_0=409600,dim_1=1024,dim_2=76800)
model_AE.load_state_dict(torch.load("/data/private/suyusheng/prompt/model/crossPromptT5_restaurant_100_t5xxl_to_t5base/68_model_cross_0.1746.pkl", map_location=lambda storage, loc:storage))

a = torch.load("tweetevalsentimentPromptXXLT5/task_prompt")
a_ = torch.load("tweetevalsentimentPromptT5/task_prompt")
a1 = torch.load("tweetevalsentimentPromptRoberta/task_prompt")
b = torch.load("restaurantPromptXXLT5/task_prompt")
b_ = torch.load("restaurantPromptT5/task_prompt")
b1 = torch.load("restaurantPromptRoberta/task_prompt")

print(a)
print("---")
print(a_)
print("---")
print(a1)
print("---")
print(model_AE(a.reshape(409600).cpu()))
print("===")
print(b)
print("---")
print(b_)
print("---")
print(b1)
print("---")
print(model_AE(b.reshape(409600).cpu()))
print("===")
