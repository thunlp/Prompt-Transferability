import torch

a = torch.load("ethicsjusticePromptRobertaLarge/1_task_prompt.pkl")
a["optimizer"]["state"][0]["step"]=0
a["trained_epoch"]=0
a["trained_step"]=0
torch.save(a,"ethicsdeontologyuseethicsjusticePromptRobertaLarge/0_task_prompt.pkl")



a = torch.load("ethicsdeontologyPromptRobertaLarge/1_task_prompt.pkl")
a["optimizer"]["state"][0]["step"]=0
a["trained_epoch"]=0
a["trained_step"]=0
torch.save(a,"ethicsjusticeuseethicsdeontologyPromptRobertaLarge/0_task_prompt.pkl")

exit()

a = torch.load("movierationalesPromptRobertaLarge/1_task_prompt.pkl")
a["optimizer"]["state"][0]["step"]=0
a["trained_epoch"]=0
a["trained_step"]=0
torch.save(a,"IMDBusemoviepPromptRobertaLarge/0_task_prompt.pkl")


a = torch.load("restaurantPromptRobertaLarge/1_task_prompt.pkl")
a["optimizer"]["state"][0]["step"]=0
a["trained_epoch"]=0
a["trained_step"]=0
torch.save(a,"laptopuserestaurantPromptRobertaLarge/0_task_prompt.pkl")


a = torch.load("snliPromptRobertaLarge/1_task_prompt.pkl")
a["optimizer"]["state"][0]["step"]=0
a["trained_epoch"]=0
a["trained_step"]=0
torch.save(a,"MNLIusesnliPromptRobertaLarge/0_task_prompt.pkl")


a = torch.load("IMDBPromptRobertaLarge/1_task_prompt.pkl")
a["optimizer"]["state"][0]["step"]=0
a["trained_epoch"]=0
a["trained_step"]=0
torch.save(a,"movierationalesuseIMDBPromptRobertaLarge/0_task_prompt.pkl")


a = torch.load("QQPPromptRobertaLarge/1_task_prompt.pkl")
a["optimizer"]["state"][0]["step"]=0
a["trained_epoch"]=0
a["trained_step"]=0
torch.save(a,"MRPCuseQQPPromptRobertaLarge/0_task_prompt.pkl")


a = torch.load("MNLIPromptRobertaLarge/1_task_prompt.pkl")
a["optimizer"]["state"][0]["step"]=0
a["trained_epoch"]=0
a["trained_step"]=0
torch.save(a,"QNLIuseMNLIPromptRobertaLarge/0_task_prompt.pkl")



a = torch.load("MRPCPromptRobertaLarge/1_task_prompt.pkl")
a["optimizer"]["state"][0]["step"]=0
a["trained_epoch"]=0
a["trained_step"]=0
torch.save(a,"QQPuseMRPCPromptRobertaLarge/0_task_prompt.pkl")


a = torch.load("laptopPromptRobertaLarge/1_task_prompt.pkl")
a["optimizer"]["state"][0]["step"]=0
a["trained_epoch"]=0
a["trained_step"]=0
torch.save(a,"restaurantuselaptopPromptRobertaLarge/0_task_prompt.pkl")


a = torch.load("MNLIPromptRobertaLarge/1_task_prompt.pkl")
a["optimizer"]["state"][0]["step"]=0
a["trained_epoch"]=0
a["trained_step"]=0
torch.save(a,"snliuseMNLIPromptRobertaLarge/0_task_prompt.pkl")


a = torch.load("IMDBPromptRobertaLarge/1_task_prompt.pkl")
a["optimizer"]["state"][0]["step"]=0
a["trained_epoch"]=0
a["trained_step"]=0
torch.save(a,"SST2useIMDBPromptRobertaLarge/0_task_prompt.pkl")


a = torch.load("restaurantPromptRobertaLarge/1_task_prompt.pkl")
a["optimizer"]["state"][0]["step"]=0
a["trained_epoch"]=0
a["trained_step"]=0
torch.save(a,"tweetuserestaurantPromptRobertaLarge/0_task_prompt.pkl")


'''
a = torch.load("/1_task_prompt.pkl")
a["optimizer"]["state"][0]["step"]=0
a["trained_epoch"]=0
a["trained_step"]=0
torch.save(a,"/0_task_prompt.pkl")
'''
