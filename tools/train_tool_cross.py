import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import shutil
from timeit import default_timer as timer
import random
import numpy as np
#from tools.eval_tool_projector import valid, gen_time_str, output_value
from tools.eval_tool import valid, gen_time_str, output_value
from tools.init_tool import init_test_dataset, init_formatter
from reader.reader import init_dataset, init_formatter, init_test_dataset
import torch.nn as nn
import torch.optim as optim
#from model.optimizer import init_optimizer
import transformers
from tools.projector import AE_0_layer, AE_1_layer_mutiple_100, AE_1_layer

logger = logging.getLogger(__name__)



def checkpoint(filename, model, optimizer, trained_epoch, config, global_step, model_AE, **kwargs):

    ####Original_model#####
    '''
    model_to_save = model.module if hasattr(model, 'module') else model
    save_params = {
        "model": model_to_save.state_dict(),
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step
    }

    try:
        torch.save(save_params, filename)
    except Exception as e:
        logger.warning("Cannot save models with error %s, continue anyway" % str(e))
    '''
    ####################

    ###model_AE
    #print("=====")
    #print(filename)
    #print("=====")
    filename = filename.strip().replace(".pkl","")
    filename = filename+"_model_cross.pkl"
    #print(filename)
    #exit()
    try:
        #torch.save(model_AE, filename)
        torch.save(model_AE.state_dict(), filename)
    except Exception as e:
        logger.warning("Cannot save models with error %s, continue anyway" % str(e))



def train(parameters, config, gpu_list, do_test=False, local_rank=-1, **params):
    epoch = config.getint("train", "epoch")
    batch_size = config.getint("train", "batch_size")

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")

    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    if os.path.exists(output_path):
        logger.warning("Output path exists, check whether need to change a name of model")
    os.makedirs(output_path, exist_ok=True)

    trained_epoch = parameters["trained_epoch"] + 1
    model = parameters["model"]




    #################################################
    ###########AE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model_AE = AE_0_layer(dim_0=768,dim_1=768).to(device)
    #model_AE = AE_0_layer(dim_0=768,dim_1=1024).to(device)
    if (config.get("model","model_size")).lower() == "large" and "base" in (params["args"].model_prompt).lower() and "100" in config.get("output","model_name"):
        #model_AE = AE_1_layer_mutiple_100(dim_0=76800,dim_1=3840,dim_2=102400).to(device)
        #model_AE = AE_1_layer_mutiple_100(dim_0=76800,dim_1=7680,dim_2=102400).to(device)
        model_AE = AE_1_layer_mutiple_100(dim_0=76800,dim_1=768,dim_2=102400).to(device)
    elif (config.get("model","model_size")).lower() == "large" and "base" in (params["args"].model_prompt).lower() and "100" not in config.get("output","model_name"):
        model_AE = AE_0_layer(dim_0=768,dim_1=1024).to(device)
    elif (config.get("model","model_size")).lower() == "base" and "base" in (params["args"].model_prompt).lower() and "100" in config.get("output","model_name"):
        ###
        #model_AE = AE_0_layer(dim_0=768,dim_1=768).to(device)
        #model_AE = AE_0_layer_76800(dim_0=76800,dim_1=76800).to(device)
        #model_AE = AE_1_layer_mutiple_100(dim_0=76800,dim_1=7680,dim_2=76800).to(device)
        model_AE = AE_1_layer_mutiple_100(dim_0=76800,dim_1=768,dim_2=76800).to(device)
        ###
    elif (config.get("model","model_size")).lower() == "base" and "base" in (params["args"].model_prompt).lower() and "100" not in config.get("output","model_name"):
        #model_AE = AE_1_layer(dim_0=768,dim_1=768,dim_2=768).to(device)
        model_AE = AE_0_layer(dim_0=768,dim_1=768).to(device)
    elif (config.get("model","model_size")).lower() == "base" and "medium" in (params["args"].model_prompt).lower():
        model_AE = AE_0_layer(dim_0=512,dim_1=768).to(device)
    elif (config.get("model","model_size")).lower() == "large":
        #Default is base
        print("-----------------------------------")
        print("Default: base to large, 768 to 1024")
        print("-----------------------------------")
        model_AE = AE_0_layer(dim_0=768,dim_1=1024).to(device)
    else:
        print("Check tool/train_tool_cross.py Line:118 AE_model")
        exit()
    #model_AE = AE_1_layer(dim_0=768,dim_1=768,dim_2=1024).to(device)
    # create an optimizer object
    # Adam optimizer with learning rate 1e-3

    #################################################
    ####Load from checkpoints and contiously training
    checkpoint_dir= "model/"+config.get("output", "model_name")

    if os.path.isdir(checkpoint_dir):
        checkpoints = os.listdir(checkpoint_dir)
        if len(checkpoints) > 0:
            print(checkpoints)
            last_checkpoint = checkpoints[0]
            for checkpoint_name in checkpoints:
                checkpoint_epoch = int(checkpoint_name.split("_")[0])
                last_checkpoint_epoch = int(last_checkpoint.split("_")[0])
                if checkpoint_epoch >= last_checkpoint_epoch:
                    last_checkpoint = checkpoint_name
            model_AE.load_state_dict(torch.load(checkpoint_dir+"/"+last_checkpoint, map_location=lambda storage, loc:storage))
        else:
            pass
    else:
        pass

    #################################################
    #################################################


    #   AdamW (
    #Parameter Group 0
    #    amsgrad: False
    #    betas: (0.9, 0.999)
    #    eps: 1e-08
    #    lr: 0.001
    #    weight_decay: 0.01
    #)
    #----
    #AdamW (
    #Parameter Group 0
    #    betas: (0.9, 0.999)
    #    correct_bias: True
    #    eps: 1e-06
    #    lr: 0.001
    #    weight_decay: 0.0
    #)


    #optimizer_AE = optim.AdamW(model_AE.parameters(), eps=1e-06, lr=1e-3, weight_decay=0.0, correct_bias=True)
    #print(model_AE)
    #print(model_AE.parameters())
    #optimizer_AE = init_optimizer(model_AE.parameters(), config, gpu_list)
    optimizer_AE = transformers.AdamW(model_AE.parameters(), eps=1e-06, lr=1e-3, weight_decay=0.0, correct_bias=True)
    #print(optimizer_AE)
    #print("----")

    #model_AE.train()
    ###########



    #optimizer = parameters["optimizer"]
    #print(optimizer)
    #print("=======")
    #exit()

    #dataset = parameters["train_dataset"]
    global_step = parameters["global_step"]
    output_function = parameters["output_function"]

    if do_test:
        init_formatter(config, ["test"])
        #test_dataset = init_test_dataset(config)


    if trained_epoch == 0:
        shutil.rmtree(
            os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")), True)

    os.makedirs(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
                exist_ok=True)

    writer = SummaryWriter(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
                           config.get("output", "model_name"))


    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "lr_multiplier")
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_AE, step_size=step_size, gamma=gamma)
    exp_lr_scheduler.step(trained_epoch)

    logger.info("Training start....")

    print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")






    #total_len = len(dataset)
    #more = ""
    #if total_len < 10000:
    #    more = "\t"
    #more = ""

    for epoch_num in range(trained_epoch, epoch):
        ###


        logger.info("Begin to initialize dataset and formatter...")
        #if mode == "train":
            #parameters["train_dataset"], parameters["valid_dataset"] = init_dataset(config, *args, **params)
        dataset, parameters["valid_dataset"] = init_dataset(config, **params)
        #else:
        #parameters["test_dataset"] = init_test_dataset(config, *args, **params)
        ###


        total_len = len(dataset)
        #print(dataset[])

        if total_len < 10000 and epoch_num==trained_epoch:
            more = "\t"


        start_time = timer()
        current_epoch = epoch_num
        #model.train()
        model.eval()
        exp_lr_scheduler.step(current_epoch)

        acc_result = None
        #acc_result_target = None
        total_loss = 0
        #total_loss_ = 0
        #total_loss_target = 0
        #total_loss_kl = 0

        output_info = ""
        step = -1

        #task_prompt = load_task_prompt()
        for step, data in enumerate(dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            ####
            #model.zero_grad()
            model_AE.zero_grad()
            ####



            results = model(data, config, gpu_list, acc_result, "train", AE=model_AE)

            loss, acc_result = results["loss"], results["acc_result"]
            #loss, loss_, loss_target, loss_kl, acc_result, acc_result_target = results["loss_total"], results["loss"], results["loss_target"], results["loss_kl"], results["acc_result"], results["acc_result_target"]




            total_loss += float(loss)
            #total_loss_ += float(loss_)
            #total_loss_target += float(loss_target)
            #total_loss_kl += float(loss_kl)

            loss.backward()
            ###AE
            #optimizer.step()
            optimizer_AE.step()

            if step % output_time == 0 and local_rank <= 0:
                output_info = output_function(acc_result, config)
                #output_info_target = output_function(acc_result_target, config)

                delta_t = timer() - start_time

                output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                             "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

                #output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                #    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                #             "%.3lf" % (total_loss_target / (step + 1)), output_info_target, '\r', config)


            global_step += 1
            writer.add_scalar(config.get("output", "model_name") + "_train_iter", float(loss), global_step)
            #writer.add_scalar(config.get("output", "model_name") + "_train_iter", float(loss_), global_step)
            #writer.add_scalar(config.get("output", "model_name") + "_train_iter", float(loss_target), global_step)
            # break
        try:
            model.module.lower_temp(0.8)
        except:
            pass

        if local_rank <= 0:
            output_info = output_function(acc_result, config)
            #output_info_target = output_function(acc_result_target, config)
            delta_t = timer() - start_time
            output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                        "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

            #output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
            #    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
            #            "%.3lf" % (total_loss_target / (step + 1)), output_info_target, None, config)

        if step == -1:
            logger.error("There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError

        if local_rank <= 0:
            #checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer, current_epoch, config, global_step, model_AE)
            checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer_AE, current_epoch, config, global_step, model_AE)
            writer.add_scalar(config.get("output", "model_name") + "_train_epoch_total_loss", float(total_loss) / (step + 1), current_epoch)
            ###
            writer.add_scalar(config.get("output", "model_name") + "_train_epoch_acc", float(acc_result['right']/acc_result['total']), current_epoch)
            ###

        if current_epoch % test_time == 0:
            with torch.no_grad():
                #####
                #valid(model, parameters["valid_dataset"], current_epoch, writer, config, gpu_list, output_function, AE=model_AE)



                #total_loss = valid(model, parameters["valid_dataset"], current_epoch, writer, config, gpu_list, output_function, AE=model_AE)
                total_loss, acc_result_eval = valid(model, parameters["valid_dataset"], current_epoch, writer, config, gpu_list, output_function, AE=model_AE)

                root_dir = "model/"+config.get("output", "model_name")
                src_checkpoint_name = root_dir+"/"+str(current_epoch)+"_model_cross.pkl"
                ###min eval loss###
                #targ_checkpoint_name = root_dir+"/"+str(current_epoch)+"_model_cross_"+str(total_loss)+".pkl"
                ###max eval acc###
                targ_checkpoint_name = root_dir+"/"+str(current_epoch)+"_model_cross_"+str(round(float(acc_result_eval['right']/acc_result_eval['total']),4))+".pkl"
                os.rename(src_checkpoint_name, targ_checkpoint_name)

                all_checkpoints = os.listdir(root_dir)
                top_5_list = list()

                for checkpoint_name in all_checkpoints:
                    if len(top_5_list) < 1:
                        top_5_list.append(checkpoint_name)
                    else:
                        ###min eval loss###
                        '''
                        for idx, in_top_5 in enumerate(top_5_list):
                            in_top_5_score = float(in_top_5.split("_")[-1].replace(".pkl",""))
                            checkpoint_score = float(checkpoint_name.split("_")[-1].replace(".pkl",""))
                            if checkpoint_score < in_top_5_score and checkpoint_name not in top_5_list:
                                #print(checkpoint_score, in_top_5_score)
                                top_5_list.insert(idx, checkpoint_name)
                            else:
                                pass
                        '''
                        ###max eval acc###
                        for idx, in_top_5 in enumerate(top_5_list):
                            in_top_5_score = float(in_top_5.split("_")[-1].replace(".pkl",""))
                            checkpoint_score = float(checkpoint_name.split("_")[-1].replace(".pkl",""))
                            if checkpoint_score > in_top_5_score and checkpoint_name not in top_5_list:
                                #print(checkpoint_score, in_top_5_score)
                                top_5_list.insert(idx, checkpoint_name)
                            else:
                                pass

                if len(top_5_list)>5:
                    top_5_list = top_5_list[:5]
                else:
                    pass

                print(top_5_list)

                if len(all_checkpoints) > 5:
                    for checkpoint_name in all_checkpoints:
                        if checkpoint_name not in top_5_list:
                            os.remove(root_dir+"/"+checkpoint_name)
                else:
                    pass
                #####

                if do_test:
                    valid(model, test_dataset, current_epoch, writer, config, gpu_list, output_function, mode="test")
        if local_rank >= 0:
            torch.distributed.barrier()
