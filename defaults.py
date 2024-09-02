from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()
# This will create a folder named `exp_name` under 'lightning_logs' to save logs, models, etc.
_C.exp_name = ""
# rand seed
_C.seed = 1
# output path
_C.outdir = "."
# GPU
_C.num_gpus = 1
# Must be pl checkpoint. This can override _C.pretrained_backbone_path.
_C.ckpt_path = ""
#progress bar
_C.enable_progress_bar = True
#fast dev for debug
_C.fast_dev_run = False
# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.train = CfgNode()

# Enable training
_C.train.enable = False

# Could be 'ddp' or 'cpu'
_C.train.strategy = "ddp"

_C.train.limit_train_batches = 1.0

_C.train.limit_val_batches = 1.0

_C.train.val_check_interval = 1.0
# save the best checkpoint by what metric?
_C.train.checkpoint_metric = ""
# "min", "max", "auto"
_C.train.checkpoint_mode = ""

# nume_workers per GPU
_C.train.num_workers = 4
# batchsize all GPU. Should be a multiple of num_gpus
_C.train.batch_size = 64

_C.train.val_only = True
# ---------------------------------------------------------------------------- #
# Validation options.
# ---------------------------------------------------------------------------- #
_C.val = CfgNode()

_C.val.val_only = False

_C.val.is_test = False

# nume_workers per GPU
_C.val.num_workers = 4
# batchsize all GPU. Should be a multiple of num_gpus
_C.val.batch_size = 64

_C.val.predict = False
# ---------------------------------------------------------------------------- #
# Test options.
# ---------------------------------------------------------------------------- #
_C.test = CfgNode()

# Enable training
_C.test.enable = False

# nume_workers per GPU
_C.test.num_workers = 4
# batchsize all GPU. Should be a multiple of num_gpus
_C.test.batch_size = 4

_C.test.limit_test_batches = 1.0

# generate logits
_C.test.gen_logits = False
# ---------------------------------------------------------------------------- #
# Solver options.
# ---------------------------------------------------------------------------- #
_C.solver = CfgNode()

_C.solver.num_epochs = 40

_C.solver.lr = 2e-4

_C.solver.weight_decay = 0.0
# learning rate policy
_C.solver.lr_policy = 'cosine_warmup'

_C.solver.warmup_epochs = 3
# optimizer
_C.solver.optimizer = "sgd"
# for SGD
_C.solver.momentum = 0.9

_C.solver.nesterov = True

_C.solver.lower_bound = 0.5
# ---------------------------------------------------------------------------- #
# Model options.
# ---------------------------------------------------------------------------- #
_C.model = CfgNode()
# how many classes to predict
_C.model.num_classes = [2, 478]
# claasification. Possibly segmentation, detection in the future
_C.model.model = "classification"
# pte, trf
_C.model.aggregator = "pte"
# mlp, multihead
_C.model.decoder = "mlp"
# ltaweightedloss,
_C.model.loss_fn = "LTAWeightedLoss"
# -1: no image feature
_C.model.img_feat_size = -1
# Image features and object features will be projected to this size.
_C.model.base_feat_size = 2048
# whether to use gt text
_C.model.text_feat_size = -1

# PTE
_C.model.pte = CfgNode()

_C.model.pte.num_heads = 8

_C.model.pte.num_layers = 3

_C.model.pte.enc_dropout = 0.1

#Llama
_C.model.llama = CfgNode()

_C.model.llama.from_pretrained = False

_C.model.llama.weight_type = 'hg'

_C.model.llama.weight_path = ''

_C.model.llama.config_path = ''

_C.model.llama.ckpt_path = ''

_C.model.llama.pos_dropout = 0.1

_C.model.llama.vocab_size = 32000

_C.model.llama.hidden_size = 768

_C.model.llama.intermediate_size = 2048

_C.model.llama.num_hidden_layers = 6

_C.model.llama.num_attention_heads = 6

_C.model.llama.num_key_value_heads = 6

_C.model.llama.max_position_embeddings = 300

_C.model.llama.tokenizer_path = ''

_C.model.llama.peft = False

_C.model.llama.quantization = True

#Llama Generation configs
_C.model.llama.generate = CfgNode()

_C.model.llama.generate.max_new_tokens = 200

_C.model.llama.generate.do_sample = True

_C.model.llama.generate.top_p = 1.0

_C.model.llama.generate.temperature = 0.7

_C.model.llama.generate.min_length = None

_C.model.llama.generate.use_cache = True

_C.model.llama.generate.top_k = 50

_C.model.llama.generate.repetition_penalty = 1.0

_C.model.llama.generate.length_penalty = 1

_C.model.llama.generate.num_return_sequences = 1

#Peft configs
_C.model.peft = CfgNode()

_C.model.peft.method = ''

_C.model.peft.r = 8

_C.model.peft.lora_alpha = 32

_C.model.peft.target_modules = ["q_proj", "v_proj"]

_C.model.peft.bias = "none"

_C.model.peft.task_type = "CAUSAL_LM"

_C.model.peft.lora_dropout = 0.05

_C.model.peft.inference_mode = False

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.data = CfgNode()
# path to annotation path
_C.data.train_anno_path = ""

_C.data.val_anno_path = ""

_C.data.test_anno_path = ""

_C.data.examples_to_keep = ""

_C.data.base_path = ""

_C.data.dataset = ""

_C.data.text = CfgNode()

_C.data.text.use = False
    
_C.data.text.max_words = 250

# -----------------------------------------------------------------------------
# Alfred specific options
# -----------------------------------------------------------------------------
_C.alfred = CfgNode()

_C.alfred.enable = False

_C.alfred.verb_only = False

_C.alfred.subgoal = 'actions'

_C.alfred.reward_label = False

_C.alfred.object_info = 'None'

_C.alfred.response_path = ''

_C.alfred.num_obj = 2

_C.alfred.use_split = 'supervised'

def get_cfg():
    return _C.clone()

