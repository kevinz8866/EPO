import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import copy
from ..utils import file_util, language_utils
from ..parser import load_config, parse_args
from operator import itemgetter
from transformers import LlamaTokenizer


class LanguageVideoDataset():
    def __init__(self, cfg, annotation_path, is_train, is_test) -> None:
        super().__init__()
        self.cfg = cfg
        self.annotation_path = annotation_path
        self.is_train = is_train
        self.is_test = is_test
        self.is_val = True if not is_train and not is_test else False
        self.annotations = self.convert(self.annotation_path)
    
    def build_tokenizer(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(self.cfg.model.llama.tokenizer_path)

    def get_label_anno(self, one_traj, traj, traj_folder):
        pass
    
    def get_text_anno(self, one_traj, traj, traj_folder):
        anno_list = []
        split = self.annotation_path.split("/")[-1]

        if self.cfg.alfred.object_info == 'None': 
            obj_text = ''
        elif self.cfg.alfred.object_info == 'interaction':
            obj_text,object_list = language_utils.get_object_text_info(one_traj)
        elif self.cfg.alfred.object_info == 'visual':
            obj_text = language_utils.get_object_visual_info(split, traj_folder, traj, one_traj)
        elif self.cfg.alfred.object_info == 'both':
            obj_text1 = language_utils.get_object_visual_info(split, traj_folder, traj, one_traj)
            obj_text2,object_list = language_utils.get_object_text_info(one_traj)
            obj_text = obj_text1 + obj_text2
        else:
            raise NotImplementedError("object info not available")

        for num, ann in enumerate(one_traj['turk_annotations']['anns']):
            #loop over the annotators
            step_count = 0
            for step_instruction in ann['high_descs']:
                #loop over the step instructions
                base_id = "{}_-_{}".format(traj_folder, traj)
                
                if not self.is_test:
                    high_actions = one_traj['plan']['high_pddl']
                    label = high_actions[step_count]['discrete_action']['action'] + " " + high_actions[step_count]['discrete_action']['args'][-1] + ' ;'
                    #label = high_actions[step_count]['discrete_action']['action'] + ' ;'
                    label = label[:-2] + " ###"
                else:
                    label = ""

                BACKGROUND = 'Based on the following instruction: ['
                QUESTION = '] What is the intended action?'

                _ = {
                    'data_id': base_id + "_-_" + f"num{num}_-_step{step_count}",
                    'prompt': obj_text + BACKGROUND + step_instruction + QUESTION + " \n\n###\n\n",
                    'completion': label
                }
                
                anno_list.append(copy.deepcopy(_))
                step_count +=1 
        return anno_list


    def convert(self, annotation_path):
        # get modalities
        modalities = []
        modalities.append('text')
        self.build_tokenizer()
        if self.cfg.data.examples_to_keep != "":
            remove_samples = True
            split = file_util.load_json(self.cfg.data.examples_to_keep)
            folder_list = list(map(itemgetter('folder'), split['supervised']))
            subfolder_list = list(map(itemgetter('subfolder'), split['supervised']))
        else:
            remove_samples = False
        
        annotations = []
        scenes = os.listdir(annotation_path)
        if self.is_test:
            traj_folder = 'xxxx__'
            for traj in scenes:
                ppp = os.path.join(annotation_path,traj,'traj_data.json')
                one_traj = file_util.load_json(ppp)
                anno = {}   # {modality_name: anno}
                for modality in modalities:
                    get_anno_func = getattr(self, f'get_{modality}_anno')
                    anno_single = get_anno_func(one_traj, traj, traj_folder)
                    if type(anno_single) == list:
                        for an in anno_single:
                            anno[modality] = an
                            annotations.append(copy.deepcopy(anno))
        else:
            for traj_folder in scenes:
                if remove_samples and (traj_folder not in folder_list):continue
                pp = os.path.join(annotation_path,traj_folder)
                trajectories = os.listdir(pp)
                for traj in trajectories:
                    if remove_samples and (traj not in subfolder_list):continue
                    ppp = os.path.join(pp,traj,'traj_data.json')
                    one_traj = file_util.load_json(ppp)
                    anno = {}   # {modality_name: anno}
                    for modality in modalities:
                        get_anno_func = getattr(self, f'get_{modality}_anno')
                        anno_single = get_anno_func(one_traj, traj, traj_folder)
                        if type(anno_single) == list:
                            for an in anno_single:
                                anno[modality] = an
                                annotations.append(copy.deepcopy(anno))      
        return annotations
    

    def fill_label(self, anno):
        pass

    def fill_text(self, anno):
        IGNORE_INDEX = -100
        PROMT_TEMPLATE = "{}"
        prompt = PROMT_TEMPLATE.format(anno["prompt"])
        example = prompt + anno["completion"]
        prompt = torch.tensor(self.tokenizer.encode(anno["prompt"]), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = IGNORE_INDEX
        max_words = self.cfg.data.text.max_words

        padding_p = max_words - prompt.shape[0]
        if padding_p > 0:
            prompt = torch.cat([torch.zeros(padding_p, dtype=torch.int64),prompt])        
            #prompt = torch.cat([prompt,torch.zeros(padding_p, dtype=torch.int64)])
        else:
            prompt = prompt[: max_words]
            print(f"wow very long prompt {padding_p}")
        prompt_mask = prompt.ne(0).float()

        padding = max_words - example.shape[0]
        if padding > 0:
            example = torch.cat([torch.zeros(padding, dtype=torch.int64),example])
            labels = torch.cat([IGNORE_INDEX*torch.ones(padding, dtype=torch.int64),labels])
        elif padding < 0:
            example = example[: max_words]
            labels = labels[: max_words]
            print(f"wow very long example {padding}")
        example_mask = example.ne(0).float()

        anno['inputs'] = example
        anno['outputs'] = labels
        anno['mask'] = example_mask
        anno['val_inputs'] = prompt
        anno['val_label'] = labels
        anno['val_mask'] = prompt_mask


    def __getitem__(self, index):
        annotation = copy.deepcopy(self.annotations[index])
        for modality in annotation:
            fill_func = getattr(self, f'fill_{modality}')
            fill_func(annotation[modality])
        item = {
            'data_id': annotation['text']['data_id'],
        }
        if not self.is_test:
            item['text'] = annotation['text']['inputs']
            item['text_annotations'] = annotation['text']['outputs']
            item['mask_text'] = annotation['text']['mask']
            if self.is_val:
                item['val_text'] = annotation['text']['val_inputs']
                item['val_annotations'] = annotation['text']['val_label']
                item['val_mask'] = annotation['text']['val_mask']
        else:
            item['val_text'] = annotation['text']['val_inputs']
            item['val_mask'] = annotation['text']['val_mask']
        return item
    

    def __len__(self):
        return len(self.annotations)
    

class AlfredDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if not hasattr(self, 'train_set'):
                self.train_set = LanguageVideoDataset(self.cfg, self.cfg.data.train_anno_path, True, False)
                print(len(self.train_set))
            if not hasattr(self, 'val_set'):
                self.val_set = LanguageVideoDataset(self.cfg, self.cfg.data.val_anno_path, False, False)
                print(len(self.val_set))

        if stage == "test" or stage is None:
            self.test_set = LanguageVideoDataset(self.cfg,self.cfg.data.test_anno_path, False, True)

    def train_dataloader(self):
        if not hasattr(self, 'train_loader'):
            num_gpus = self.cfg.num_gpus
            assert self.cfg.train.batch_size % num_gpus == 0
            batch_size = self.cfg.train.batch_size // num_gpus
            self.train_loader = DataLoader(self.train_set, shuffle=True, batch_size=batch_size, num_workers=self.cfg.train.num_workers, pin_memory=True)
        return self.train_loader

    def val_dataloader(self):
        if not hasattr(self, 'val_loader'):
            num_gpus = self.cfg.num_gpus
            assert self.cfg.val.batch_size % num_gpus == 0
            batch_size = self.cfg.val.batch_size // num_gpus
            self.val_loader = DataLoader(self.val_set, shuffle=False, batch_size=batch_size, num_workers=self.cfg.val.num_workers, pin_memory=True)
        return self.val_loader

    def test_dataloader(self):
        if not hasattr(self, 'test_loader'):
            # num_gpus = self.cfg.num_gpus
            num_gpus = 1
            assert self.cfg.test.batch_size % num_gpus == 0
            batch_size = self.cfg.test.batch_size // num_gpus
            self.test_loader = DataLoader(self.test_set, shuffle=False, batch_size=batch_size, num_workers=self.cfg.test.num_workers, drop_last=False, pin_memory=True)
        return self.test_loader
