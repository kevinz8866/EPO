import torch
from torch import nn
from torch.distributions import Categorical
import math
from ..parser import parse_args, load_config
from pytorch_lightning import seed_everything
from transformers import (
    LlamaForSequenceClassification,
    LlamaForCausalLM,
    LlamaConfig,
)
from peft import get_peft_model, prepare_model_for_int8_training, set_peft_model_state_dict
from ..utils import peft_utils, file_util


class LlamaText(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        llama_config = self.build_llama_config()
        if self.cfg.model.llama.from_pretrained:
            if self.cfg.model.llama.weight_type == 'pt':
                sd = torch.load(self.cfg.model.llama.weight_path)
                llama = LlamaForCausalLM(llama_config)
                llama.load_state_dict(sd,strict=False)
            elif self.cfg.model.llama.weight_type == 'hg':
                llama = LlamaForCausalLM.from_pretrained(
                    self.cfg.model.llama.weight_path,
                    load_in_8bit= True if self.cfg.model.llama.quantization else None,
                    device_map="auto" if self.cfg.model.llama.quantization else None,
                )
            elif self.cfg.model.llama.weight_type == 'ckpt':
                llama = LlamaForCausalLM.from_pretrained(
                    self.cfg.model.llama.weight_path,
                    load_in_8bit= True if self.cfg.model.llama.quantization else None,
                    device_map="auto" if self.cfg.model.llama.quantization else None,
                )
        else:
            llama = LlamaForCausalLM(llama_config)
        
        if self.cfg.model.peft.method != '':
            if self.cfg.model.llama.quantization:
                llama = prepare_model_for_int8_training(llama)
            peft_config = peft_utils.generate_peft_config(self.cfg)
            self.llama = get_peft_model(llama, peft_config) #implement other peft methods configs
            self.llama.print_trainable_parameters()
            #print(self.llama.device)
        else:
            self.llama = llama
        
        if self.cfg.model.llama.from_pretrained and self.cfg.model.llama.weight_type == 'ckpt':
            ckpt = torch.load(self.cfg.model.llama.ckpt_path)
            new_state_dict = {k.replace('model.llama.','').replace('default.',''): v for k, v in ckpt['state_dict'].items()}
            set_peft_model_state_dict(self.llama, new_state_dict)
        
    def build_llama_config(self):

        if self.cfg.model.llama.config_path != '':
            config = file_util.load_json(self.cfg.model.llama.config_path)
            llama_config = LlamaConfig(**config)
            return llama_config
        
        llama_config = LlamaConfig(
                            vocab_size= self.cfg.model.llama.vocab_size,
                            hidden_size= self.cfg.model.llama.hidden_size,
                            intermediate_size= self.cfg.model.llama.intermediate_size,
                            num_hidden_layers= self.cfg.model.llama.num_hidden_layers,
                            num_attention_heads= self.cfg.model.llama.num_attention_heads,
                            num_key_value_heads= self.cfg.model.llama.num_key_value_heads,
                            max_position_embeddings= self.cfg.model.llama.max_position_embeddings,
                            )
        return llama_config

    def forward(self, batch):
        outputs = self.llama(**batch)
        return outputs 
    
    def generate(self, batch, tokenizer):
        outputs = self.llama.generate(
            **batch,
            max_new_tokens=self.cfg.model.llama.generate.max_new_tokens,
            do_sample=self.cfg.model.llama.generate.do_sample,
            top_p=self.cfg.model.llama.generate.top_p,
            temperature=self.cfg.model.llama.generate.temperature,
            min_length=self.cfg.model.llama.generate.min_length,
            use_cache=self.cfg.model.llama.generate.use_cache,
            top_k=self.cfg.model.llama.generate.top_k,
            repetition_penalty=self.cfg.model.llama.generate.repetition_penalty,
            length_penalty=self.cfg.model.llama.generate.length_penalty,
            num_return_sequences = self.cfg.model.llama.generate.num_return_sequences,
            pad_token_id = tokenizer.eos_token_id
        )
        return outputs 


class LlamaHead(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        llama_config = self.build_llama_config()
        if self.cfg.model.llama.from_pretrained:
            if self.cfg.model.llama.weight_type == 'pt':
                sd = torch.load(self.cfg.model.llama.weight_path)
                llama = LlamaForSequenceClassification(llama_config)
                llama.load_state_dict(sd,strict=False)
            elif self.cfg.model.llama.weight_type == 'hg':
                llama = LlamaForSequenceClassification.from_pretrained(
                    self.cfg.model.llama.weight_path,
                    load_in_8bit= True if self.cfg.model.llama.quantization else None,
                    device_map="auto" if self.cfg.model.llama.quantization else None,
                    num_labels=self.cfg.model.num_classes[0] 
                )
            elif self.cfg.model.llama.weight_type == 'ckpt':
                llama = LlamaForSequenceClassification.from_pretrained(
                    self.cfg.model.llama.weight_path,
                    load_in_8bit= True if self.cfg.model.llama.quantization else None,
                    device_map="auto" if self.cfg.model.llama.quantization else None,
                    num_labels=self.cfg.model.num_classes[0] 
                )
        else:
            raise NotImplementedError
        
        if self.cfg.model.peft.method != '':
            if self.cfg.model.llama.quantization:
                llama = prepare_model_for_int8_training(llama)
            peft_config = peft_utils.generate_peft_config(self.cfg)
            self.llama = get_peft_model(llama, peft_config) #implement other peft methods configs
            self.llama.print_trainable_parameters()
            #print(self.llama.device)
        else:
            self.llama = llama
        
        if self.cfg.model.llama.from_pretrained and self.cfg.model.llama.weight_type == 'ckpt':
            ckpt = torch.load(self.cfg.model.llama.ckpt_path)
            new_state_dict = {k.replace('model.llama.','').replace('default.',''): v for k, v in ckpt['state_dict'].items()}
            set_peft_model_state_dict(self.llama, new_state_dict)

    def build_llama_config(self):

        if self.cfg.model.llama.config_path != '':
            config = file_util.load_json(self.cfg.model.llama.config_path)
            llama_config = LlamaConfig(**config)
            return llama_config
        
        kwargs = {"num_labels": self.cfg.model.num_classes[0]}
        llama_config = LlamaConfig(
                            vocab_size = self.cfg.model.llama.vocab_size,
                            hidden_size = self.cfg.model.base_feat_size,
                            intermediate_size = self.cfg.model.llama.intermediate_size,
                            num_hidden_layers = self.cfg.model.llama.num_hidden_layers,
                            num_attention_heads = self.cfg.model.llama.num_attention_heads,
                            num_key_value_heads = self.cfg.model.llama.num_key_value_heads,
                            max_position_embeddings = self.cfg.model.llama.max_position_embeddings,
                            **kwargs
                            )
        return llama_config

    def forward(self, batch):
        outputs = self.llama(**batch)
        return outputs

# --------------------------------------------------------------------#


class MLPDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.head = nn.Linear(cfg.model.base_feat_size, sum(cfg.model.num_classes))
    
    def forward(self, x):
        # x: (B, Z, D)
        logits = self.head(x)  # (B, Z, #verbs + #nouns)
        logits = torch.split(logits, self.cfg.model.num_classes, dim=-1)  # [(B, Z, #verbs), (B, Z, #nouns)]
        return logits  
    
# --------------------------------------------------------------------#

def sanity_check():
    args = parse_args()
    cfg = load_config(args)
    
if __name__ == '__main__':
    sanity_check()
