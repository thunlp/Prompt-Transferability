import torch
import torch.nn as nn


class PromptEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.prompt_embeddings = nn.Embedding(config.prompt_len, config.hidden_size)
        self.prompt_embeddings.weight.data.uniform_(-0.5, 0.5)
        # self.prompt_embeddings.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def init_prompt_emb(self, init_ids, word_embeddings):
        prompt_weights = word_embeddings(init_ids).detach()
        self.prompt_embeddings.weight.data = nn.Parameter(prompt_weights)

    def prepend_prompt_embeddings(self, inputs_embeds, word_embeddings):
        '''Prepend prompt embeddings'''

        batch_size, seq_length, _ = inputs_embeds.shape

        mask_embed = word_embeddings(torch.tensor([self.config.mask_token_id]).repeat(batch_size, 1).to(inputs_embeds.device))
        prompt_ids = torch.arange(self.config.prompt_len).repeat(batch_size, 1).to(inputs_embeds.device)
        prompt_embed = self.prompt_embeddings(prompt_ids)
        inputs_embeds = torch.cat([mask_embed, prompt_embed, inputs_embeds], dim=1)
        
        return inputs_embeds

    def prepend_attention_mask(self, attention_mask, token_type_ids=None):
        '''Prepend attention_mask and token_type_ids for prompt'''

        batch_size, seq_length = attention_mask.shape
        extended_seq_length = seq_length + self.config.prompt_len + 1

        prompt_attention_mask = torch.ones(self.config.prompt_len + 1).repeat(batch_size, 1).to(attention_mask.device)
        attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1).long()

        if token_type_ids is not None:
            prompt_token_type_ids = torch.ones(self.config.prompt_len + 1).repeat(batch_size, 1).to(token_type_ids.device)
            token_type_ids = torch.cat([prompt_token_type_ids, token_type_ids], dim=1).long()
        
        else:
            token_type_ids = torch.zeros(extended_seq_length, dtype=torch.long, device=attention_mask.device)
        
        return attention_mask, token_type_ids

    def prepend_position_embeddings(self, position_embeddings):
        '''Prepend position embeddings for prompt'''

        if self.config.position_embedding_type == 'all_zero':
            prompt_position_embeddings = torch.zeros((1, self.config.prompt_len + 1, self.config.hidden_size)).to(input_ids.device)
            position_embeddings = torch.cat([prompt_position_embeddings, position_embeddings], dim=1)
        
        elif self.config.position_embedding_type == 'absolute':
            total_len = self.config.prompt_len + 1 + position_embeddings.shape[1]
            position_ids = torch.arange(total_len).to(position_embeddings.device)
            position_embeddings = self.position_embeddings(position_ids)

        else:
            raise NotImplementedError

        return position_embeddings

class PromptModel(nn.Module):
    def __init__(self, args):
        raise NotImplementedError

    def freeze_lm(self):
        for n, p in self.named_parameters():
            if 'prompt' in n:
                p.requires_grad = True
                print(n, p.requires_grad)
            else:
                p.requires_grad = False

    def init_prompt_emb(self, prompt, init_ids):
        if init_ids is not None:
            self.backbone.bert.embeddings.init_prompt_emb(torch.tensor(init_ids, dtype=torch.long))

        else:
            init.xavier_uniform(prompt)

    def forward(self):
        raise NotImplementedError
