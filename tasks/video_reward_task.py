import os
import torch
import numpy as np
from .video_task import VideoTask
from ..utils import eval_util, file_util, language_utils
import torch.nn.functional as F
from transformers import LlamaTokenizer

class VideoRewardTask(VideoTask):
    def __init__(self, cfg, steps_in_epoch):
        super().__init__(cfg, steps_in_epoch)
        self.cfg = cfg
        self.build_tokenizer()
        self.model.llama.config.pad_token_id = 0

    def build_tokenizer(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(self.cfg.model.llama.tokenizer_path)

    def forward(self, batch, train=True):
        input_texts = batch['text']
        output_texts = batch['text_annotations']
        mask_text = batch['mask_text']
        input_batch = {
            "input_ids": input_texts,
            "labels": output_texts,
            "attention_mask":mask_text,
        }
        
        outputs = self.model.forward(input_batch) 
        return outputs
    
    def predict(self, batch, train=False):
        input_texts = batch['text']
        mask_text = batch['mask_text'] 
        input_batch = {
            "input_ids": input_texts,
            "attention_mask":mask_text,
        }
        outputs = self.model.forward(input_batch) 
        return outputs

    def training_step(self, batch, batch_idx): 
        step_results = {}
        outputs = self.forward(batch,True)
        step_results['loss'] = outputs.loss
        step_results['train/loss'] = outputs.loss.item()
        self.log('train/loss_step', outputs.loss.item(), rank_zero_only=True)
        
        target_label = batch['text_annotations']
        if target_label is not None:

            top1_acc, top5_acc = eval_util.distributed_topk_accs(outputs.logits, target_label, (1, 5), None)
            
        step_results['train/top1_acc'] = top1_acc
        step_results['train/top5_acc'] = top5_acc
        return step_results


    def training_epoch_end(self, outputs):
        keys = [x for x in outputs[0].keys() if x != "loss"]
        for key in keys:
            metric = sum([x[key] for x in outputs]) / len(outputs)
            self.log(key, metric, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        step_results = {}
        if self.cfg.val.predict:
            to_save = {}
            outputs = self.predict(batch,False)
            input_text = self.tokenizer.batch_decode(batch['text'], skip_special_tokens=True)
            probs = F.softmax(outputs.logits[:, :2], dim=-1).tolist()
            for i,j,k in zip(batch['data_id'], input_text, probs):
                to_save.update({i:(j,k)})
            return to_save
        
        outputs = self.forward(batch,False)
        step_results['val/loss'] = outputs.loss.item()
        self.log('val/loss_step', outputs.loss.item(), rank_zero_only=True)
        target_label = batch['text_annotations']
        if target_label is not None:
            top1_acc, top5_acc = eval_util.distributed_topk_accs(outputs.logits, target_label, (1, 5), None)
            print()
            print(f"token acc: {top1_acc}")
            print()
        step_results['val/top1_acc'] = top1_acc
        return step_results


    def validation_epoch_end(self, outputs):
        if self.cfg.val.predict:
            to_save = {}
            for x in outputs:
                to_save.update(x)

            base_dir = self.trainer.log_dir
            save_dir = os.path.join(base_dir, 'submit.json')
            file_util.save_json(to_save, save_dir)
            return

        keys = [x for x in outputs[0].keys()]
        for key in keys:
            metric = sum([x[key] for x in outputs]) / len(outputs)
            self.log(key, metric, sync_dist=True)
            print(f"epoch token acc: {metric}")
