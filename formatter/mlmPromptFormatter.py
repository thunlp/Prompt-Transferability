from transformers import AutoTokenizer
import torch
import json
import numpy as np
from .Basic import BasicFormatter
import random
import logging

class mlmPromptFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.max_len = config.getint("train", "max_len")
        self.prompt_len = config.getint("prompt", "prompt_len")
        self.prompt_num = config.getint("prompt", "prompt_num")
        self.mode = mode
        ##########
        self.model_base = config.get("model","model_base")
        self.model_name = config.get("output","model_name")
        if "Roberta" in self.model_base:
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        elif "Bert" in self.model_base:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            print("Have no matching in the formatter")
            print("MLM")
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        #self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        ##########
        self.prompt_prefix = [- (i + 1) for i in range(self.prompt_len)]
        # self.prompt_middle = [- (i + 1 + self.prompt_len) for i in range(self.prompt_len)]

    def process(self, data, config, mode, *args, **params):
        if params["args"]["args"].pre_train_mlm==True:
            return self.convert_example_to_features(data)


    def convert_example_to_features(self, data):

        inputx = []
        mask = []
        label = []

        max_len = self.max_len + 3 + self.prompt_num

        #input_ids, lm_label_ids, input_mask,
        for d in data:
            #tokens = self.tokenizer.encode(d["sent"], add_special_tokens = False)
            if "sent1" in d and "sent2" in d:
                for sent in ["sent1","sent2"]:
                    input_ids = self.tokenizer.encode(d[sent], add_special_tokens = False)
                    ################
                    ################
                    if len(input_ids) > self.max_len-3:
                        input_ids = input_ids[:self.max_len-3]

                    input_ids, lm_label_ids = self.random_word(input_ids)

                    input_ids = self.prompt_prefix + [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
                    lm_label_ids = [-100]*len(self.prompt_prefix) + [-100]*len([self.tokenizer.cls_token_id]) + lm_label_ids + [-100]*len([self.tokenizer.sep_token_id])

                    input_mask = [1] * len(input_ids)

                    while len(input_ids) < max_len:
                        input_ids.append(0)
                        input_mask.append(0)
                        #segment_ids.append(0)
                        lm_label_ids.append(-100)

                    inputx.append(input_ids)
                    mask.append(input_mask)
                    label.append(lm_label_ids)


            else:
                input_ids = self.tokenizer.encode(d["sent"], add_special_tokens = False)
            #input_ids = self.tokenizer.encode(d["sent"], add_special_tokens = False)
                ################
                ################
                if len(input_ids) > self.max_len-3:
                    input_ids = input_ids[:self.max_len-3]

                input_ids, lm_label_ids = self.random_word(input_ids)

                input_ids = self.prompt_prefix + [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
                lm_label_ids = [-100]*len(self.prompt_prefix) + [-100]*len([self.tokenizer.cls_token_id]) + lm_label_ids + [-100]*len([self.tokenizer.sep_token_id])

                input_mask = [1] * len(input_ids)

                while len(input_ids) < max_len:
                    input_ids.append(0)
                    input_mask.append(0)
                    #segment_ids.append(0)
                    lm_label_ids.append(-100)

                inputx.append(input_ids)
                mask.append(input_mask)
                label.append(lm_label_ids)


        ret = {
            "inputx": torch.tensor(inputx, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
        }

        return ret


    def random_word(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of str, tokenized sentence.
        :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
        :return: (list of str, list of int), masked tokens and related labels for LM prediction
        """
        output_label = []

        #tokens --> token id
        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to [mask] token
                #print(self.tokenizer.decode[50264])
                if prob < 0.8:
                    if "Roberta" in self.model_base:
                        tokens[i] = self.tokenizer.encode("<mask>",add_special_tokens=False)[0]
                    elif "Bert" in self.model_base:
                        tokens[i] = self.tokenizer.encode("[MASK]",add_special_tokens=False)[0]
                    else:
                        print("Wrong!!")
                        print("Wrong!!")
                        print("Wrong!!")
                        print("replace with Roberta")
                        print("MLM")
                        tokens[i] = self.tokenizer.encode("<mask>",add_special_tokens=False)[0]

                    #tokens[i] = "[MASK]"

                # 10% randomly change token to random token
                '''
                elif prob < 0.9:
                    #tokens[i] = random.choice(list(self.tokenizer.vocab.items()))[0]
                    tokens[i] = random.choice(list(self.tokenizer.vocab.items()))[1]

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                '''
                try:
                    #output_label.append(self.tokenizer.vocab[token])
                    output_label.append(token)
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    ###
                    if "Roberta" in self.model_base:
                        output_label.append(self.tokenizer.vocab["<unk>"])
                    elif "Bert" in self.model_base:
                        output_label.append(self.tokenizer.vocab["[UNK]"])
                    else:
                        print("Wrong!!")
                        print("Wrong!!")
                        print("Wrong!!")
                        print("Replace with Roberta")
                        print("MLM")
                        output_label.append(self.tokenizer.vocab["<unk>"])
                    ###
                    #output_label.append(self.tokenizer.vocab["[UNK]"])
                    logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-100)

        return tokens, output_label


