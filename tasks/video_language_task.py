import os
import torch
import numpy as np
from .video_task import VideoTask
from ..utils import eval_util, file_util, language_utils
import re
from transformers import LlamaTokenizer

class VideoLanguageTask(VideoTask):
    def __init__(self, cfg, steps_in_epoch):
        super().__init__(cfg, steps_in_epoch)
        self.cfg = cfg
        self.build_tokenizer()

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

    def generate(self, batch, train=False):
        input_texts = batch['val_text']
        mask_text = batch['val_mask']
        input_batch = {
            "input_ids": input_texts,
            "attention_mask": mask_text
        }
        outputs = self.model.generate(input_batch, self.tokenizer)
        return outputs

    def training_step(self, batch, batch_idx): 
        step_results = {}
        outputs = self.forward(batch,True)
        step_results['loss'] = outputs.loss
        step_results['train/loss'] = outputs.loss.item()
        self.log('train/loss_step', outputs.loss.item(), rank_zero_only=True)
        target_texts = batch['text_annotations'] if self.cfg.data.text.use else None
        if target_texts is not None:
            mask = (target_texts != -100)
            shifted_logits = torch.roll(outputs.logits, shifts=1, dims=1)
            top1_acc, top5_acc = eval_util.distributed_topk_accs(shifted_logits, target_texts, (1, 5), mask)
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
        dummy_loss = 0.0
        step_results['val/loss'] = dummy_loss
        self.log('val/loss_step', dummy_loss, rank_zero_only=True)
        outputs = self.generate(batch,False)

        to_save = {}
        if self.cfg.val.predict:
            output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for i,j in zip(batch['data_id'], output_text):
                to_save.update({i:j})
            return to_save
        
        input_texts = batch['val_text']
        target_texts = batch['val_annotations']

        outputs = outputs[:, input_texts.shape[1]:]
        mask = (target_texts != 0)

        _, current_size = outputs.size()
        if current_size < target_texts.size(1):
            padding_size = target_texts.size(1) - current_size
            outputs = torch.nn.functional.pad(outputs, (0, padding_size))
        elif current_size > target_texts.size(1):
            outputs = outputs[:, :target_texts.size(1)]

        top1_acc, top5_acc = eval_util.distributed_topk_accs(outputs, target_texts, (1, 5), mask)
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
