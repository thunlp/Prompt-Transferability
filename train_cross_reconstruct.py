import torch
from torch import nn
import os
import argparse
import random
from config_parser import create_config
from torch import optim
from tools.projector import AE_1_layer as AE




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", default='config/train_task_projection_reconstructionLoss.config')
    parser.add_argument('--gpu', '-g', help="gpu id list", default='0')
    parser.add_argument("--target_model", type=str, default="Roberta")
    parser.add_argument("--mlm", default=False, action="store_true")

    args = parser.parse_args()
    configFilePath = args.config
    config = create_config(configFilePath)

    ####
    if args.mlm:
        output = "model/cross_Bert_to_Roberta_reconstructionLoss_mlm"
    else:
        output = "model/cross_Bert_to_Roberta_reconstructionLoss_only_imdb_laptop"


    if os.path.isdir(output):
        pass
    else:
        os.mkdir(output)
    ####

    all_dir = os.listdir("task_prompt_emb")
    if args.mlm:
        all_dir = [dir for dir in all_dir if "_mlm_s1" in dir or "_mlm_s2" in dir]
    else:
        all_dir = [dir for dir in all_dir if "_mlm_" not in dir]
    print("---")
    print(all_dir)
    print("---")
    #exit()

    given_task = config.get("dataset","dataset").split(",")

    prompt_dataset_model = {d:torch.load("task_prompt_emb/"+d+"/task_prompt") for d in all_dir}

    if args.mlm:
        dataset = all_dir
    else:
        dataset = list(set([d.split("Prompt")[0] for d in all_dir if d.split("Prompt")[0] in given_task]))
    model_type = {"Roberta","Bert"}
    print(dataset)
    #exit()


    ####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model_AE = AE(input_dim=76800,compress_dim=3).to(device)
    model_AE = AE(input_dim=76800,compress_dim=768).to(device)
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
                if args.mlm:
                    input_ten = torch.stack([prompt_dataset_model[dataset] for dataset in choosed_dataset]).to(device)
                    input_ten = input_ten.reshape(input_ten.shape[0],int(input_ten.shape[1])*int(input_ten.shape[2])).to(device)
                    target_ten = torch.stack([prompt_dataset_model[dataset] for dataset in choosed_dataset]).to(device)
                    target_ten = target_ten.reshape(target_ten.shape[0],int(target_ten.shape[1])*int(target_ten.shape[2])).to(device)
                else:
                    input_ten = torch.stack([prompt_dataset_model[dataset+"PromptBert"] for dataset in choosed_dataset]).to(device)
                    input_ten = input_ten.reshape(input_ten.shape[0],int(input_ten.shape[1])*int(input_ten.shape[2])).to(device)
                    target_ten = torch.stack([prompt_dataset_model[dataset+"PromptRoberta"] for dataset in choosed_dataset]).to(device)
                    target_ten = target_ten.reshape(target_ten.shape[0],int(target_ten.shape[1])*int(target_ten.shape[2])).to(device)
            else:
                if args.mlm:
                    input_ten = torch.stack([prompt_dataset_model[dataset] for dataset in choosed_dataset]).to(device)
                    input_ten = input_ten.reshape(input_ten.shape[0],int(input_ten.shape[1])*int(input_ten.shape[2])).to(device)
                    target_ten = torch.stack([prompt_dataset_model[dataset] for dataset in choosed_dataset]).to(device)
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
        #print(model_AE.state_dict())
        torch.save(model_AE.state_dict(), str(output+"/"+str(epoch)+"_loss:"+str("{:.3f}".format(float(loss.data)))+"_model.pkl"))



