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
from tools.eval_tool_projector import valid, gen_time_str, output_value
from tools.init_tool import init_test_dataset, init_formatter
from reader.reader import init_dataset, init_formatter, init_test_dataset
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)



class AE(nn.Module):
    def __init__(self, **kwargs):
        super(AE, self).__init__()
        self.encoder = nn.Linear(
            in_features=kwargs["input_dim"], out_features=int(kwargs["input_dim"]/100)
        )
        self.decoder = nn.Linear(
            in_features=int(kwargs["input_dim"]/100), out_features=kwargs["input_dim"]
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
        torch.save(model_AE, filename)
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



    optimizer = parameters["optimizer"]

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
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    exp_lr_scheduler.step(trained_epoch)

    logger.info("Training start....")

    print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")


    ###########AE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_AE = AE(input_dim=76800,compress_dim=3).to(device)
    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer_AE = optim.Adam(model_AE.parameters(), lr=1e-3)
    model_AE.train()
    ###########


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
        total_loss = 0

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




            total_loss += float(loss)

            loss.backward()
            ###AE
            #optimizer.step()
            optimizer_AE.step()

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
            checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer, current_epoch, config, global_step, model_AE)
            writer.add_scalar(config.get("output", "model_name") + "_train_epoch", float(total_loss) / (step + 1), current_epoch)

        if current_epoch % test_time == 0:
            with torch.no_grad():
                ###
                valid(model, parameters["valid_dataset"], current_epoch, writer, config, gpu_list, output_function, AE=model_AE)
                ###
                if do_test:
                    valid(model, test_dataset, current_epoch, writer, config, gpu_list, output_function, mode="test")
        if local_rank >= 0:
            torch.distributed.barrier()
