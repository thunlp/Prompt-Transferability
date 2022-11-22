import torch
import torch.nn as nn
from typing import *
from openprompt.prompts import SoftTemplate
from openprompt.data_utils import InputExample, InputFeatures


class CrossModelProjector(SoftTemplate):
    def __init__(self, args, source_config, target_config, num_layers=10, flatten=True, **kwargs):
        super().__init__(**kwargs)

        self.args = args
        self.soft_embeds.requires_grad = False

        if flatten:
            in_dim = args.prompt_len * source_config.hidden_size
            hidden_dim = args.prompt_len * target_config.hidden_size
            out_dim = args.prompt_len * target_config.hidden_size

        else:
            in_dim = source_config.hidden_size
            hidden_dim = target_config.hidden_size
            out_dim = target_config.hidden_size

        hidden_dim = 768
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 768),
            nn.LeakyReLU(),
            nn.Linear(768, out_dim)
        )

        # i_d = in_dim
        # o_d = hidden_dim
        # proj = nn.ModuleList()
        # for i in range(num_layers):
        #     proj.append(nn.Linear(i_d, o_d))
        #     proj.append(nn.LeakyReLU())
        #     i_d = o_d
        #     o_d = out_dim
        
        # self.proj = nn.Sequential(proj)

    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        inputs_embeds = self.raw_embedding(batch['input_ids'])
        batch_size = inputs_embeds.size(0)
        if self.num_tokens > 0:
            soft_embeds = self.soft_embeds.flatten()
            soft_embeds = self.proj(soft_embeds)
            soft_embeds = soft_embeds.reshape((self.args.prompt_len, -1))
            soft_embeds = soft_embeds.repeat(batch_size, 1, 1)
            inputs_embeds = torch.cat([soft_embeds, inputs_embeds], 1)

        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        if 'attention_mask' in batch and self.num_tokens>0:
            am = batch['attention_mask']
            batch['attention_mask'] = torch.cat([torch.ones((batch_size,self.num_tokens), dtype = am.dtype,device=am.device), am], dim=-1)
        return batch

    @torch.no_grad()
    def get_cross_model_prompt(self):
        return self.proj(self.soft_embeds)
