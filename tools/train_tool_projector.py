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
from tools.eval_tool import valid, gen_time_str, output_value
from tools.init_tool import init_test_dataset, init_formatter

logger = logging.getLogger(__name__)


def checkpoint(filename, model, optimizer, trained_epoch, config, global_step):
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


def train(parameters, config, gpu_list, do_test=False, local_rank=-1):
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
    optimizer = parameters["optimizer"]
    global_step = parameters["global_step"]
    output_function = parameters["output_function"]
    #dataset = parameters["train_dataset"]
    ###
    #print(config.sections())
    #print(config.get("data","train_dataset_type"))
    #dataset_all = [{"train_dataset_"+str(dataset):parameters["train_dataset_"+str(dataset)]} for dataset in config.get("data","train_dataset_type").strip().split(",")]

    dataset_all={}
    for dataset in config.get("data","train_dataset_type").strip().split(","):
        dataset_all["train_dataset_"+str(dataset)] = parameters["train_dataset_"+str(dataset)]
    print(dataset_all)
    ###

    '''
    if do_test:
        init_formatter(config, ["test"])
        test_dataset = init_test_dataset(config)
    '''

    if trained_epoch == 0:
        shutil.rmtree(
            os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")), True)

    os.makedirs(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
                exist_ok=True)

    writer = SummaryWriter(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
                           config.get("output", "model_name"))

    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "lr_multiplier")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    exp_lr_scheduler.step(trained_epoch)

    logger.info("Training start....")


    print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")


    #total_len = len(dataset)
    total_len = min([len(v) for key,v in dataset_all.items()])
    total_len_max = min([len(v) for key,v in dataset_all.items()])

    print("Min Iteration {}".format(total_len))
    print("Max Iteration {}".format(total_len_max))
    print("Default Batch Size {}".format(config.get("train","batch_size")))
    print("Actual training Batch Size {}".format(max(int(int(config.get("train","batch_size"))/len(dataset_all.keys())),1)*len(dataset_all.keys())))


    ##################Make all input same dim##############
    ##Find max length
    print(all_dataset)
    max_length=0
    for idx, dataset in enumerate(all_dataset):
        #print("====")
        if mode == "train":
            for line in result["train_dataset_"+str(dataset)]:
                max_length = max(int(line['inputx'].shape[-1]), max_length)
                break
        else:
            for line in result["test_dataset_"+str(dataset)]:
                max_length = max(int(line['inputx'].shape[-1]), max_length)
                break
    print("max_length: {}".format(max_length))

    max_length = max_length+100
    ##Alter to same length
    for idx, dataset in enumerate(all_dataset):
        #print(result["train_dataset_"+str(dataset)])
        #exit()
        if mode == "train":
            for line in result["train_dataset_"+str(dataset)]:
                #print(line)
                #print("===========")
                #print(line['inputx'].shape)
                pad_id_input = torch.ones( int(line['inputx'].shape[0]), max_length-int(line['inputx'].shape[-1]), dtype=int)
                pad_id_mask = torch.zeros(int(line['mask'].shape[0]),max_length-int(line['mask'].shape[-1]))

                result["train_dataset_"+str(dataset)]["inputx"] = torch.stack([result["train_dataset_"+str(dataset)]["inputx"],pad_id_input],dim=1)
                result["train_dataset_"+str(dataset)]["mask"] = torch.stack([result["train_dataset_"+str(dataset)]["mask"],pad_id_mask],dim=1)
                #print("-----------")
                #print(line['inputx'].shape)
                break
    '''
    #print(result["train_dataset_"+str("IMDB")])
    for line in result["train_dataset_"+str("IMDB")]:
        print(line)
        break
    for line in result["train_dataset_"+str("laptop")]:
        print(line)
        break
    exit()
    '''
    #######################################################


    more = ""
    if total_len < 10000:
        more = "\t"

    for epoch_num in range(trained_epoch, epoch):
        start_time = timer()
        current_epoch = epoch_num
        model.train()
        exp_lr_scheduler.step(current_epoch)

        acc_result = None
        total_loss = 0

        output_info = ""
        step = -1

        ###
        #dataset = dataset_all['train_dataset_IMDB']
        #print("-----")
        #print(dataset)
        #print("-----")
        ###

        for step, data in enumerate(dataset):
            #print("========")
            #print(data.keys())
            #print("========")
            #exit()
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            model.zero_grad()

            results = model(data, config, gpu_list, acc_result, "train")

            loss, acc_result = results["loss"], results["acc_result"]
            total_loss += float(loss)

            loss.backward()
            optimizer.step()

            if step % output_time == 0 and local_rank <= 0:
                output_info = output_function(acc_result, config)

                delta_t = timer() - start_time

                output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                             "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

            global_step += 1
            writer.add_scalar(config.get("output", "model_name") + "_train_iter", float(loss), global_step)
            # break
        try:
            model.module.lower_temp(0.8)
        except:
            pass

        if local_rank <= 0:
            output_info = output_function(acc_result, config)
            delta_t = timer() - start_time
            output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                        "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

        if step == -1:
            logger.error("There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError

        if local_rank <= 0:
            checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer, current_epoch, config, global_step)
            writer.add_scalar(config.get("output", "model_name") + "_train_epoch", float(total_loss) / (step + 1), current_epoch)

        '''
        if current_epoch % test_time == 0:
            with torch.no_grad():
                valid(model, parameters["valid_dataset"], current_epoch, writer, config, gpu_list, output_function)
                if do_test:
                    valid(model, test_dataset, current_epoch, writer, config, gpu_list, output_function, mode="test")
        '''
        if local_rank >= 0:
            torch.distributed.barrier()
