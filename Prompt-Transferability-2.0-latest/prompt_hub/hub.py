import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from openprompt.utils.reproduciblity import set_seed
from openprompt import PromptDataLoader, PromptForClassification
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from datasets import load_dataset, load_metric
from openprompt.plms import load_plm
from openprompt.prompts import SoftTemplate, ManualVerbalizer

from .data_processor import data_processor_list
from .utils import decorate


class PromptHub(Trainer):
    def __init__(self, prompt_emb=None, **kwargs):
        args = kwargs['args']
        self.out_dir_root = args.output_dir

        processor = data_processor_list[args.dataset]()

        # Model
        template_text = '{"soft": None, "duplicate": ' + str(args.prompt_len) + ', "same": True} {"mask"} {"placeholder": "text_a"} {"placeholder": "text_b"}'
        model, template, verbalizer, plm, tokenizer, model_config, tokenizer_wrapper_class, model_type = self.get_model(args.backbone, processor, args)
        self.set_active_state_dict(model)   # Only save soft prompts

        # Initialize transformers.trainer
        kwargs['model'] = model
        kwargs['tokenizer'] = tokenizer
        kwargs['train_dataset'] = processor.train_dataset
        kwargs['eval_dataset'] = processor.eval_dataset
        super().__init__(**kwargs)
    
        self.config = model_config
        self.plm = plm
        self.template_text = template_text
        self.template = template
        self.verbalizer = verbalizer
        self.tokenizer_wrapper_class = tokenizer_wrapper_class
        self.prompt_emb = None

        # Transfer
        self.source_model_type = model_type
        self.target_model_type = None


        print('Trainable parameters:')
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                print(n, p.shape)

        print(f'Template: {self.template}')
        print(f'Verbalizer: {self.verbalizer}')
        print(f'Raw input example: {self.train_dataset[0]}')
        print(f'Wrapped input example: {self.template.wrap_one_example(self.train_dataset[0])}')

    def get_model(self, model_name, processor, args):
        if 'bert-' in model_name:
            model_type = 'bert'

        elif 'roberta-' in model_name:
            model_type = 'roberta'

        elif 't5-' in model_name:
            model_type = 't5'

        elif 'gpt' in model_name:
            model_type = 'gpt'

        # Load openprompt models
        if (not hasattr(self, 'args')) or args.backbone != model_name:
            plm, tokenizer, model_config, tokenizer_wrapper_class = load_plm(model_type, model_name)
        else:
            plm = self.plm
            tokenizer = self.tokenizer
            model_config = self.config
            tokenizer_wrapper_class = self.tokenizer_wrapper_class
            model_type = self.source_model_type

        # Load soft template
        if hasattr(self, 'template') and self.template is not None:
            template = self.template
        else:
            template_text = '{"soft": None, "duplicate": ' + str(args.prompt_len) + ', "same": True} {"mask"} {"placeholder": "text_a"} {"placeholder": "text_b"}'
            template = SoftTemplate(model=plm, text=template_text, tokenizer=tokenizer, num_tokens=args.prompt_len) # initialize_from_vocab=args.init_from_vocab
        
        # Load verbalizer
        if hasattr(self, 'verbalizer') and self.verbalizer is not None:
            verbalizer = self.verbalizer
        else:
            verbalizer = ManualVerbalizer(tokenizer, classes=processor.labels, label_words=processor.label_words)
        
        if hasattr(self, 'model') and self.model is not None:
            model = self.model
        else:
            model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer, freeze_plm=True)

            if hasattr(args, "model_parallel") and args.model_parallel:
                print('parallelize model!')
                model.parallelize()

        _keys_to_ignore_on_save = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                _keys_to_ignore_on_save.append(n)

        model._keys_to_ignore_on_save = _keys_to_ignore_on_save

        return model, template, verbalizer, plm, tokenizer, model_config, tokenizer_wrapper_class, model_type

    def compute_loss(self, model, inputs, return_outputs=False):        
        outputs = model(inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = nn.CrossEntropyLoss()(outputs, inputs['label'])

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        labels = nested_detach(tuple(inputs.__getitem__(name) for name in ['label']))
        if len(labels) == 1:
            labels = labels[0]

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, logits = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def _prepare_input(self, data):
        kwargs = dict(device=self.args.device)
        return data.to(**kwargs)
    
    def _prepare_inputs(self, inputs):
        inputs = self._prepare_input(inputs)
        return inputs
        
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataloader = PromptDataLoader(
            dataset=self.train_dataset, 
            template=self.template, 
            tokenizer=self.tokenizer,
            tokenizer_wrapper_class=self.tokenizer_wrapper_class, 
            batch_size=self._train_batch_size,
            max_seq_length=self.args.max_source_length, 
            decoder_max_length=1,
            shuffle=True)

        return train_dataloader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        validation_dataloader = PromptDataLoader(
            dataset=eval_dataset, 
            template=self.template, 
            tokenizer=self.tokenizer,
            tokenizer_wrapper_class=self.tokenizer_wrapper_class, 
            batch_size=self.args.per_device_eval_batch_size,
            max_seq_length=self.args.max_source_length, 
            decoder_max_length=1,
            shuffle=False) 

        return validation_dataloader

    @torch.no_grad()
    def get_prompt_emb(self):

        if 'bert-' in self.args.backbone:
            prompt_emb = self.backbone.bert.embeddings.prompt_embeddings.weight

        if 'roberta-' in self.args.backbone:
            prompt_emb = self.backbone.roberta.embeddings.prompt_embeddings.weight

        if 't5-' in self.args.backbone:
            prompt_emb = self.backbone.encoder.embeddings.prompt_embeddings.weight

        return prompt_emb.detach().cpu()

    def train_prompt(self, model=None, task=None, **kwargs):
        self.args.output_dir = os.path.join(self.out_dir_root, 'prompt_emb')
        os.makedirs(self.args.output_dir, exist_ok=True)

        if model is None:
            model = self.model
        
        else:
            if isinstance(model, str):
                if model != self.args.backbone:
                    processor = data_processor_list[task]
                    model, template, verbalizer, plm, tokenizer, model_config, tokenizer_wrapper_class, model_type = self.get_model(model, processor, self.args)
                else:
                    model = self.model

            elif isinstance(model, torch.nn.Module):
                pass
        
        self._move_model_to_device(model, self.args.device)

        if task != self.args.dataset:
            processor = data_processor_list[self.args.dataset]()
            self.train_dataset = processor.train_dataset

        return super().train(**kwargs)
        
    def eval_prompt(self, model=None, eval_dataset=None, prompt_emb=None):
        r"""Evaluate a prompt"""

        self.args.output_dir = os.path.join(self.out_dir_root, 'prompt_emb')
        os.makedirs(self.args.output_dir, exist_ok=True)

        if model is not None and isinstance(model, str) and model != self.args.backbone:
            model, template, verbalizer, plm, tokenizer, model_config, tokenizer_wrapper_class, model_type = self.get_model(args.backbone, processor, args)
            self.model = model

        if prompt_emb is not None:
            if isinstance(prompt_emb, str):
                prompt_emb = torch.load(prompt_emb, map_location='cpu')
            elif isinstance(prompt_emb, torch.Tensor):
                pass
            else:
                raise NotImplementedError

            self.backbone.roberta.embeddings = nn.Parameter(prompt_emb.detach())

        if eval_dataset is None or eval_dataset == self.args.dataset:                    # Use default
            eval_dataset = self.eval_dataset

        elif eval_dataset != self.args.dataset:     # Use a dataset different from the source dataset
            processor = data_processor_list[eval_dataset]()
            eval_dataset = processor.eval_dataset

        else:
            raise NotImplementedError

        metrics = self.evaluate(eval_dataset=eval_dataset)
        self.save_metrics("eval_prompt", metrics)

        return metrics

    def cross_task_eval(self, model=None, target_task=None, prompt_emb=None):
        r"""Performs cross-task transfer evaluation."""

        self.args.output_dir = os.path.join(self.out_dir_root, 'prompt_emb')
        os.makedirs(self.args.output_dir, exist_ok=True)

        metrics = self.eval_prompt(model=model, eval_dataset=target_task, prompt_emb=prompt_emb)
        self.save_metrics("cross_task_eval", metrics)

        return metrics

    def cross_model_train(self, source_model=None, target_model=None, task=None, prompt_emb=None, **kwargs):
        r"""Performs corss-model transfer."""

        from .cross_model import CrossModelProjector

        self.args.output_dir = os.path.join(self.out_dir_root, 'cross_model_projector')
        os.makedirs(self.args.output_dir, exist_ok=True)

        processor = data_processor_list[task]()
        _, _, verbalizer, plm, tokenizer, target_model_config, tokenizer_wrapper_class, model_type = self.get_model(target_model, processor, self.args)
        template = CrossModelProjector(
            args=self.args, source_config=self.config, target_config=target_model_config, 
            num_layers=self.args.num_proj_layers, flatten=self.args.flatten_proj,
            model=plm, text=self.template_text, tokenizer=tokenizer, num_tokens=self.args.prompt_len)
        self.model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer, freeze_plm=True)
        _keys_to_ignore_on_save = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                _keys_to_ignore_on_save.append(n)

        self.model._keys_to_ignore_on_save = _keys_to_ignore_on_save

        # Load source prompt or specific prompt
        # if prompt_emb is None:
        #     prompt_emb = os.path.join(self.out_dir_root, 'prompt_emb/checkpoint-711776/pytorch_model.bin')
        
        if isinstance(prompt_emb, str):
            prompt_emb = torch.load(prompt_emb, map_location='cpu')['prompt_model.template.soft_embeds'].to(self.args.device)
            self.model.prompt_model.template.soft_embeds = nn.Parameter(prompt_emb, requires_grad=False)
        elif isinstance(prompt_emb, torch.Tensor):
            self.model.prompt_model.template.soft_embeds = nn.Parameter(prompt_emb, requires_grad=False)

        self._move_model_to_device(self.model, self.args.device)

        trainable_parameter_names = []
        print('Trainable parameters:')
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                print(n, p.shape)
                trainable_parameter_names.append(n)

        self.set_active_state_dict(self.model, trainable_parameter_names)

        if task != self.args.dataset:
            self.train_dataset = processor.train_dataset
            self.eval_dataset = processor.eval_dataset

        self.args.learning_rate = 0.1    
        train_result = super().train(**kwargs)
        
        return train_result

    def cross_model_eval(self, source_model=None, target_model=None, task=None, prompt_emb=None):
        self.args.output_dir = os.path.join(self.out_dir_root, 'cross_model_projector')
        os.makedirs(self.args.output_dir, exist_ok=True)

        if task != self.args.dataset:
            processor = data_processor_list[task]()
            eval_dataset = processor.eval_dataset
        else:
            eval_dataset = self.eval_dataset

        metrics = self.evaluate(eval_dataset=eval_dataset)
        self.save_metrics("eval_prompt", metrics)

        return metrics

    def activated_neuron(self, model=None, task=None, layers=None, prompt_emb=None):
        r"""Get activated neuron."""

        processor = data_processor_list[task]()

        model, _, _, _, tokenizer, model_config, tokenizer_wrapper_class, model_type = self.get_model(model, processor, self.args)
        self._move_model_to_device(model, self.args.device)
        num_layers = model_config.num_hidden_layers
        
        if prompt_emb is not None:
            self.model.prompt_model.template.soft_embeds = nn.Parameter(prompt_emb, requires_grad=False)

        from openprompt.data_utils.utils import InputExample
        data = [InputExample(guid=0, text_a='<s>')]
        loader = PromptDataLoader(
            dataset=data, 
            template=self.template, 
            tokenizer=self.tokenizer,
            tokenizer_wrapper_class=self.tokenizer_wrapper_class, 
            batch_size=1,
            max_seq_length=self.args.max_source_length, 
            decoder_max_length=1,
            shuffle=False)

        outputs = [[] for _ in range(num_layers)]
        def save_ppt_outputs1_hook(n):
            def fn(_,__,output):
                outputs[n].append(output.detach().to('cpu'))
            return fn

        for n in range(num_layers):
            if model_type == 'bert' or model_type == 'roberta':
                model.prompt_model.plm.roberta.encoder.layer[n].intermediate.register_forward_hook(save_ppt_outputs1_hook(n))
            elif model_type == 't5':
                model.prompt_model.plm.t5.decoder.block[n].layer[2].DenseReluDense.wi.register_forward_hook(save_ppt_outputs1_hook(n))
            elif model_type == 'gpt':
                model.prompt_model.plm.gpt.transformer.h[n].mlp.c_fc.register_forward_hook(save_ppt_outputs1_hook(n))
            else:
                raise NotImplementedError

        for inputs in loader:
            _ = model(inputs.to(self.args.device))

        for k in range(num_layers):
            outputs[k] = torch.cat(outputs[k])

        outputs = torch.stack(outputs)
        outputs = outputs[:,:1,:1,:]    # Get the output of the <mask> position

        if layers is not None:
            outputs = outputs[torch.tensor(layers)]
            num_layers = len(layers)
        
        outputs = outputs.view(num_layers, -1)

        # Active neuron before ReLU
        torch.save(outputs, f'outputs/{self.args.backbone}/{self.args.dataset}_{self.args.seed}/activated_neuron_before_relu.pt')

        # Active neuron after ReLU
        neuron_after_relu = (outputs > 0).int()
        torch.save(neuron_after_relu, f'outputs/{self.args.backbone}/{self.args.dataset}_{self.args.seed}/activated_neuron_after_relu.pt')

        return outputs, neuron_after_relu

    def mask_activated_neuron(self, model=None, task=None, layers=None, ratio=0.2):
        
        processor = data_processor_list[task]()

        model, _, _, _, tokenizer, model_config, tokenizer_wrapper_class, model_type = self.get_model(model, processor, self.args)
        self._move_model_to_device(model, self.args.device)
        num_layers = model_config.num_hidden_layers

        neuron = torch.load(f'outputs/{self.args.backbone}/{self.args.dataset}_{self.args.seed}/activated_neuron_before_relu.pt')
        original_shape = neuron.shape
        neuron = neuron.reshape(-1)
        mask = torch.ones_like(neuron)
        idx = torch.argsort(neuron, descending=True)
        idx = idx[:int(ratio * len(idx))]
        mask[idx] = 0
        mask = mask.view(original_shape)

        def save_ppt_outputs1_hook(n):
            def fn(_,__,output):
                output = output * mask[n].to('cuda')
                return output
            return fn

        if layers is None:
            layers = range(num_layers)

        for n in layers:
            if model_type == 'bert' or model_type == 'roberta':
                model.prompt_model.plm.roberta.encoder.layer[n].intermediate.register_forward_hook(save_ppt_outputs1_hook(n))
            elif model_type == 't5':
                model.prompt_model.plm.t5.decoder.block[n].layer[2].DenseReluDense.wi.register_forward_hook(save_ppt_outputs1_hook(n))
            elif model_type == 'gpt':
                model.prompt_model.plm.gpt.transformer.h[n].mlp.c_fc.register_forward_hook(save_ppt_outputs1_hook(n))
            else:
                raise NotImplementedError

        eval_results = self.eval_prompt(model=model, eval_dataset=task)

        return eval_results, mask
        
    def plot_neuron(self, model=None, task=None, **kwargs):
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(40, 20))
        neuron = torch.load(f'outputs/{self.args.backbone}/{self.args.dataset}_{self.args.seed}/activated_neuron_before_relu.pt')
        sns.heatmap(neuron.numpy(), cmap='Reds', **kwargs)
        plt.xlabel('Neuron')
        plt.ylabel('Layer')

    def set_active_state_dict(self, module: nn.Module, includes=['prompt_model.template.soft_embeds']):
        r"""modify the state_dict function of the model (by default, the backbone model) to return only the tunable part.
        Args:
            module (:obj:`nn.Module`): The module modified. The modification is in-place.
        """
        
        def _caller(_org_func, includes,  *args, **kwargs):
            state_dict = _org_func(*args, **kwargs)
            keys = list(state_dict.keys())
            for n  in keys:
                if n not in includes:
                    state_dict.pop(n)
            return state_dict
        includes = includes          # use excludes will have trouble when the model have shared weights
        if hasattr(module.state_dict, "__wrapped__"):
            raise RuntimeWarning("The forward function might have been wrapped by a decorator, is it intended? Do you freeze the parameters twice?")
        module.state_dict = decorate(module.state_dict, _caller, extras=(includes,), kwsyntax=True)
