import torch

a = torch.load("laptopPromptRoberta/1_task_prompt.pkl")
#{'model': tensor([[-0.1109,  0.0176, -0.1202,  ..., -0.0485,  0.0071,  0.2300],
b = torch.load("../tools/laptop/task_prompt")
print(a)
#print(a["model"])
#print("----")
#print(b)
#print(type(b))
#exit()
#'param_groups': [{'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-06, 'weight_decay': 0.0, 'correct_bias': True, 'initial_lr': 0.001, 'params': [0]}]}, 'trained_epoch': 1, 'global_step': 147}

a["model"]=b
a["optimizer"]["state"][0]["step"]=0
a["trained_epoch"]=0
a["trained_step"]=0
#print("====")
#print(a)
#exit()

torch.save(a,"laptop_base_emotionPromptRoberta_tanh/0_task_prompt.pkl")
