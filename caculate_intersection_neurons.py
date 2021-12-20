import os
import torch


#relu = torch.nn.ReLU()



root = "task_activated_neuron_xxlarge"

#wi0
#wi1

dirs = os.listdir(root)

#print(dirs)

for dir in dirs:
    #names = ["IMDB","MRPC","QQP","justice","movie","nqopen","samsum","squad","tweet","MNLI","QNLI","deont","laptop","multinews","restaurant","snli","sst-2"]
    names = ["MRPC","laptop","multinews","restaurant","samsum"]

    for name in names:
        task_ten_1_neurons = torch.load(root+"/neurons/wi0/"+name+"/neurons.pt", map_location=lambda storage, loc: storage)

        task_ten_2_neurons = torch.load(root+"/neurons_new/wi0/"+name+"/neurons.pt", map_location=lambda storage, loc: storage)


        task_ten_1_neurons[task_ten_1_neurons>0] = float(1)
        task_ten_1_neurons[task_ten_1_neurons<=0] = float(0)

        task_ten_2_neurons[task_ten_2_neurons>0] = float(1)
        task_ten_2_neurons[task_ten_2_neurons<=0] = float(0)


        _sum = task_ten_1_neurons + task_ten_2_neurons

        #and
        #_and = torch.dot(task_ten_1_neurons, task_ten_2_neurons)
        _and = float(len(_sum[_sum>=2]))

        #or
        #_or = task_ten_1_neurons + task_ten_2_neurons
        #_or[_or>=1] = float(1)
        #_or = torch.FloatTensor([[ 0.1133, -0.9567,  0.2958]])
        #print(_or)
        #print(len(_or[_or>0]))
        #_or = torch.sum(_or)
        #print(_or)
        #exit()
        _or = float(len(_sum[_sum>0]))

        print(name, _and, "/", _or)
        print(name, "{:.2f}".format((100*_and/_or)))
        print("----------------------------------")
    print("======================================")
    print("======================================")

