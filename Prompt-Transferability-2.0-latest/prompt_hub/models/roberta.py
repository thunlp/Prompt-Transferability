from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.utils import logging
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaEmbeddings,
    RobertaForMaskedLM,
    RobertaLMHead,
    create_position_ids_from_input_ids
)

from .base import PromptEmbeddings

logger = logging.get_logger(__name__)


class PromptRobertaEmbeddings(PromptEmbeddings, RobertaEmbeddings):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # input_ids: [<mask>, prompt_0, prompt_1, ..., prompt_L-1, token_0, token_1, ..., token_N-1]
            # L: prompt length
            # N: sentence length

            # Sentence embeddings
            select_word = (input_ids >= 0).int()
            word_embeds = self.word_embeddings(input_ids * select_word) * select_word.unsqueeze(2)

            # Prompt embeddings
            select_prompt = (input_ids < 0).int()
            prompt_ids = - (input_ids + 1)
            prompt_embs = self.prompt_embeddings(prompt_ids * select_prompt) * select_prompt.unsqueeze(2)

            # Sum to get input_embed
            inputs_embeds = word_embeds + prompt_embs
            inputs_embeds = self.prepend_prompt_embeddings(inputs_embeds, self.word_embeddings)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            position_embeddings = self.prepend_position_embeddings(position_embeddings)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class PromptRoberta(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = RobertaForMaskedLM.from_pretrained(args.backbone)
        self.config = self.backbone.config
        self.num_labels = self.config.num_labels

        for k, v in vars(args).items():
            setattr(self.config, k, v)

        self.backbone.roberta.embeddings = PromptRobertaEmbeddings(self.config)

        for n, p in self.named_parameters():
            if 'prompt' in n:
                p.requires_grad = True
                print(n, p.requires_grad)
            else:
                p.requires_grad = False

        self.config.mask_token_id = 50264

        self.init_prompt_emb([50264] * self.config.prompt_len)

    def init_prompt_emb(self, init_ids):
        self.backbone.roberta.embeddings.init_prompt_emb(torch.tensor(init_ids, dtype=torch.long), self.backbone.roberta.embeddings.word_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        verbalizer: dict = {0: 4997, 1: 3893}
    ):

        verbalizer = {0: 10932, 1: 2362}
        # Extend attention_mask and token_type_ids
        attention_mask, token_type_ids = self.backbone.roberta.embeddings.prepend_attention_mask(attention_mask, token_type_ids)
        
        lm_output = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )['logits']                         # [batch_size, seq_length, vocab_size]
        mask_logits = lm_output[:, 0]       # [batch_size, vocab_size], select the logits of <mask>, which is the 1st token

        # Select scores based on verbalizer
        # logits = []
        # for label_mapped_token_id in verbalizer.values():
        #     logits.append(mask_logits[:, label_mapped_token_id].unsqueeze(1))
        logits = torch.cat([mask_logits[:, 4997].unsqueeze(1), mask_logits[:, 3893].unsqueeze(1)], dim=1)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return loss, logits
