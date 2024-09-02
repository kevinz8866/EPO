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
        
        obj_text, object_list = language_utils.get_object_text_info(one_traj)

        for num, ann in enumerate(one_traj['turk_annotations']['anns']):
            #loop over the annotators
            if num != 0: continue
            step_count = 0
            for step_instruction in ann['high_descs']:
                #loop over the step instructions
                base_id = "{}_-_{}".format(traj_folder, traj)
                high_actions = one_traj['plan']['high_pddl']
                true_action = high_actions[step_count]['discrete_action']['action']
                true_obj = high_actions[step_count]['discrete_action']['args'][-1]
                BACKGROUND = 'Based on the following instruction: ['
                QUESTION = '] What is the intended action?'
                text = obj_text + BACKGROUND + step_instruction + QUESTION 
                
                bad_obj = 0
                object_list = [true_obj] + object_list
                for obj_num, obj in enumerate(object_list):
                    if true_action == 'GotoLocation': break
                    if bad_obj > self.cfg.alfred.num_obj: break
                    prompt = text + f" Proposed answer: {true_action + ' ' + obj}"
                    if obj == true_obj:
                        label = 1
                    else:
                        label = 0
                        bad_obj += 1
                    _ = {
                        'data_id': base_id + "_-_" + f"num{num}_-_step{step_count}" + "_-_" + f"num{obj_num}",
                        'prompt': prompt + " \n\n###\n\n",
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
            split_data = file_util.load_json(self.cfg.data.examples_to_keep)
            folder_list = list(map(itemgetter('folder'), split_data[self.cfg.alfred.use_split]))
            subfolder_list = list(map(itemgetter('subfolder'), split_data[self.cfg.alfred.use_split]))
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
        if self.cfg.val.val_only:
            IGNORE_INDEX = -100
            PROMT_TEMPLATE = "{}"
            prompt = PROMT_TEMPLATE.format(anno["prompt"])
            prompt = torch.tensor(
                self.tokenizer.encode(anno["prompt"]), dtype=torch.int64
            )
            max_words = self.cfg.data.text.max_words
            padding_p = max_words - prompt.shape[0]    
            if padding_p > 0:
                prompt = torch.cat([torch.zeros(padding_p, dtype=torch.int64),prompt])
            elif padding_p <0:
                prompt = prompt[: max_words]

            prompt_mask = prompt.ne(0)
            prompt_mask = prompt_mask.float()
            anno['val_inputs'] = prompt
            anno['val_mask'] = prompt_mask
        else:
            IGNORE_INDEX = -100
            PROMT_TEMPLATE = "{}"
            prompt = PROMT_TEMPLATE.format(anno["prompt"])
            prompt = torch.tensor(
                self.tokenizer.encode(anno["prompt"]), dtype=torch.int64
            )
            labels = torch.tensor(
                anno["completion"], dtype=torch.int64
            )
            max_words = self.cfg.data.text.max_words

            padding_p = max_words - prompt.shape[0]    
            if padding_p > 0:
                prompt = torch.cat([torch.zeros(padding_p, dtype=torch.int64),prompt])
            elif padding_p <0:
                prompt = prompt[: max_words]
                print("wow very long prompt")
            
            prompt_mask = prompt.ne(0)
            prompt_mask = prompt_mask.float()
            anno['inputs'] = prompt
            anno['outputs'] = labels
            anno['mask'] = prompt_mask


    def __getitem__(self, index):
        annotation = copy.deepcopy(self.annotations[index])
        for modality in annotation:
            fill_func = getattr(self, f'fill_{modality}')
            fill_func(annotation[modality])

        item = {
            'data_id': annotation['text']['data_id'],
        }
        if not self.cfg.val.val_only:
            item['text'] = annotation['text']['inputs']
            item['text_annotations'] = annotation['text']['outputs']
            item['mask_text'] = annotation['text']['mask']
        else:
            item['text'] = annotation['text']['val_inputs']
            item['mask_text'] = annotation['text']['val_mask']

        return item

    

    def __len__(self):
        return len(self.annotations)
    

class AlfredRewardModule(pl.LightningDataModule):
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
