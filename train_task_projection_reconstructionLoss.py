import torch
from torch import nn
import os
import argparse
import random
from config_parser import create_config
from torch import optim



class AE(nn.Module):
    def __init__(self, **kwargs):
        super(AE, self).__init__()
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





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", default='config/train_task_projection_reconstructionLoss.config')
    parser.add_argument('--gpu', '-g', help="gpu id list", default='0')
    parser.add_argument("--target_model", type=str, default="Roberta")

    args = parser.parse_args()
    configFilePath = args.config
    config = create_config(configFilePath)


    all_dir = os.listdir("task_prompt_emb")
    all_dir = [dir for dir in all_dir if "_mlm_" not in dir]
    #print(all_dir)

    given_task = config.get("dataset","dataset").split(",")


    prompt_dataset_model = {d:torch.load("task_prompt_emb/"+d+"/task_prompt") for d in all_dir}

    dataset = list(set([d.split("Prompt")[0] for d in all_dir if d.split("Prompt")[0] in given_task]))
    model_type = {"Roberta","Bert"}
    #print(dataset)
    #exit()


    ####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_AE = AE(input_dim=76800,compress_dim=3).to(device)
    optimizer_AE = optim.Adam(model_AE.parameters(), lr=1e-5)
    loss_fun = nn.MSELoss()
    model_AE.train()
    ####


    for epoch in range(int(config.get("train","epoch"))):
        for iter in range(500):
            try:
                choosed_dataset = random.sample(dataset,k=int(config.get("train","batch_size")))
            except:
                choosed_dataset = random.sample(dataset,k=int(len(dataset)))
                choosed_dataset += random.choices(dataset,k=int(config.get("train","batch_size"))-len(dataset))


            if args.target_model == "Roberta":
                input_ten = torch.stack([prompt_dataset_model[dataset+"PromptBert"] for dataset in choosed_dataset]).to(device)
                input_ten = input_ten.reshape(input_ten.shape[0],int(input_ten.shape[1])*int(input_ten.shape[2])).to(device)
                target_ten = torch.stack([prompt_dataset_model[dataset+"PromptRoberta"] for dataset in choosed_dataset]).to(device)
                target_ten = target_ten.reshape(target_ten.shape[0],int(target_ten.shape[1])*int(target_ten.shape[2])).to(device)
            else:
                input_ten = torch.stack([prompt_dataset_model[dataset+"PromptRoberta"] for dataset in choosed_dataset]).to(device)
                input_ten = input_ten.reshape(input_ten.shape[0],int(input_ten.shape[1])*int(input_ten.shape[2])).to(device)
                target_ten = torch.stack([prompt_dataset_model[dataset+"PromptBert"] for dataset in choosed_dataset])
                target_ten = target_ten.reshape(target_ten.shape[0],int(target_ten.shape[1])*int(target_ten.shape[2])).to(device)


            optimizer_AE.zero_grad()

            output_ten = model_AE(input_ten)
            loss = loss_fun(output_ten, target_ten)
            loss.backward()
            optimizer_AE.step()

        print("Epoch:",epoch,"iter:",iter,"loss:",loss)

        torch.save(model_AE, "model/train_task_projection_reconstructionLoss/"+str(epoch)+"_loss:"+str("{:.2f}".format(float(loss.data)))+"_model")



