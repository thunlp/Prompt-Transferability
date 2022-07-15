import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaPreTrainedModel, 
    RobertaModel, 
    RobertaEmbeddings, 
    create_position_ids_from_input_ids,
    RobertaLMHead
)
from transformers.activations import ACT2FN, gelu
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.modeling_outputs import SequenceClassifierOutput, MaskedLMOutput


class RobertaForMaskedLMPrompt(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, prompt_len=100, init_ids=None):
        super().__init__(config)

        self.prompt_len = prompt_len

        self.roberta = RobertaModelWarp(config, prompt_len, init_ids, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

        self.freeze_lm()

    def freeze_lm(self):
        for name, param in self.named_parameters():
            if not 'prompt_embedding' in name:
                param.requires_grad = False

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mask_id=50264,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        ## Add <MASK> to input_ids
        mask_ids = torch.tensor([mask_id]).repeat(batch_size, 1).to(input_ids.device)
        input_ids = torch.cat([mask_ids, input_ids], dim=1)
        
        ## Add prefix to attention_mask
        prompt_attention_mask = torch.ones(self.prompt_len + 1).repeat(batch_size, 1).to(attention_mask.device)
        attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)

        outputs = self.roberta(
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
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        # "positive":22173, "negative":33407
        # "yes":10932, "no":2362
        # '1': 134, '-': 12
        # 'true': 29225, 'false': 22303
        mask_logits = prediction_scores[:, 0, :]
        logits = torch.cat([mask_logits[:, 33407].unsqueeze(1), mask_logits[:, 22173].unsqueeze(1)], dim=1)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )

class RobertaLMHeadWarp(nn.Module):
    """WARP Roberta LM Head without last decoder layer."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        return x

class RobertaWarp(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, prompt_len=100, init_ids=None):
        super().__init__(config)

        self.prompt_len = prompt_len

        self.roberta = RobertaModelWarp(config, prompt_len, init_ids, add_pooling_layer=False)
        self.lm_decoder = RobertaLMHeadWarp(config)
        self.label_embedding = nn.Linear(config.hidden_size, config.num_labels)
        self.sigm = nn.Sigmoid()

        self.init_weights()

        self.freeze_lm()

    def freeze_lm(self):
        for name, param in self.named_parameters():
            if not ('prompt_embedding' in name or 'label_embedding' in name):
                param.requires_grad = False

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mask_id=50264,
    ):
        """
            mask_id: token ID of <mask>
        """

        batch_size, seq_length = input_ids.shape

        ## Add <MASK> to input_ids
        mask_ids = torch.tensor([mask_id]).repeat(batch_size, 1).to(input_ids.device)
        input_ids = torch.cat([mask_ids, input_ids], dim=1)
        
        ## Add prefix to attention_mask
        prompt_attention_mask = torch.ones(self.prompt_len + 1).repeat(batch_size, 1).to(attention_mask.device)
        attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        
        outputs = self.roberta(
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
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state[:, 0, :]

        lm_score = self.lm_decoder(hidden_states)
        logits = self.label_embedding(lm_score)
        # logits = self.sigm(logits)

        # print('shape', outputs.last_hidden_state.shape, mask.shape, hidden_states.shape, lm_score.shape, logits.shape, labels.shape)

        masked_lm_loss = None
        if labels is not None:
            # label_emb = self.label_embedding(labels)
            # loss_fct = nn.BCELoss()
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits.view(-1, self.config.num_labels), labels)

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
        )
    
    def tie_weights(self):
        """No need to tie input and outpu embeddings."""
        pass

class RobertaEmbeddingsWarp(RobertaEmbeddings):
    """
    Difference from Hugginface source code: 
        Add prompt embedding in constructor
        Update forward pass
    """

    def __init__(self, config, prompt_len, init_ids=None):
        super().__init__(config)

        ## Different from hugginface source code
        self.prompt_len = prompt_len
        self.prompt_embedding = nn.Embedding(prompt_len, config.hidden_size)

        self.init_prompt_embedding(input_ids=init_ids)

    def init_prompt_embedding(self, input_ids=None, normal_params=None):
        if input_ids is not None or normal_params is not None:
            if input_ids is not None:
                embedding = self.word_embeddings(input_ids)

            elif normal_params is not None:
                mean = self.word_embeddings.mean()
                std = self.word_embeddings.std()
                embedding = torch.normal(mean, std)
        
            self.prompt_embedding.weight = nn.Parameter(embedding)

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        batch_size, seq_length = input_ids.shape
        
        ## Create position_ids for prompt
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                input_ids_prefix = torch.zeros((self.prompt_len)).repeat(batch_size, 1).to(input_ids.device) + 10 # make the prompt ids != pad id
                position_ids = create_position_ids_from_input_ids(
                    torch.cat([input_ids_prefix, input_ids], dim=1), self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

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
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        ## This is the original RoBERTa embedding, without positional embedding
        embeddings = inputs_embeds + token_type_embeddings

        ## Add prefix
        prompt_ids = torch.arange(self.prompt_len).repeat(batch_size, 1).to(input_ids.device)
        prompt_embedding = self.prompt_embedding(prompt_ids)

        embeddings = torch.cat([embeddings[:, :1, :], prompt_embedding, embeddings[:, 1:, :]], dim=1)

        ## Then add positional embedding
        if self.position_embedding_type == "absolute":

            ## Add prefix for position embedding
            position_ids = torch.arange(seq_length + self.prompt_len).repeat(batch_size, 1).to(input_ids.device)

            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class RobertaModelWarp(RobertaModel):
    """
    Difference from Huggingface source code: Use new embedding layer instead of original RobertaEmbeddings
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, prompt_len, init_ids=None, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddingsWarp(config, prompt_len, init_ids)
