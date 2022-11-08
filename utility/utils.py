import logging
import os
import sys
import torch
import numpy as np
import argparse
import random
from contextlib import contextmanager
import torch.distributed as dist

from transformers import AutoTokenizer, AutoConfig, AutoModel
from nltk.tokenize import wordpunct_tokenize
import json
from models.tnlrv3.modeling import TuringNLRv3ForSequenceClassification
from models.tnlrv3.configuration_tnlrv3 import TuringNLRv3Config
from models.tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer
from models.fast import Fastformer
MODEL_CLASSES = {
    # 'unilm': (TuringNLRv3Config, Fastformer, TuringNLRv3Tokenizer),
    'unilm': (TuringNLRv3Config, TuringNLRv3ForSequenceClassification, TuringNLRv3Tokenizer),
    'others': (AutoConfig, AutoModel, AutoTokenizer)
}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def setuplogger(log_path="/workspaceblobstore/v-wenjunpeng/mind/log.txt"):
    root = logging.getLogger()
    # logging.basicConfig(format="[%(levelname)s %(asctime)s] %(message)s", level=logging.INFO)
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    if (root.hasHandlers()):
        root.handlers.clear()
    root.addHandler(handler)

    fileHandler = logging.FileHandler(log_path)
    root.addHandler(fileHandler)


def dump_args(args):
    for arg in dir(args):
        if not arg.startswith("_"):
            logging.info(f"args[{arg}]={getattr(args, arg)}")

def init_process(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12365'

    # initialize the process group
    os.environ["RANK"] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, )
    torch.cuda.set_device(rank)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


def cleanup_process():
    dist.destroy_process_group()


def get_device():
    if torch.cuda.is_available():
        local_rank = os.environ.get("RANK", 0)
        return torch.device('cuda', int(local_rank))
    return torch.device('cpu')


def get_barrier(dist_training):
    if dist_training:
        return dist.barrier

    def do_nothing():
        pass

    return do_nothing


@contextmanager
def only_on_main_process(local_rank, barrier):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    need = True
    if local_rank not in [-1, 0]:
        barrier()
        need = False
    yield need
    if local_rank == 0:
        barrier()

def warmup_linear(args, step):
    if step <= args.warmup_step:
        return step/args.warmup_step
    return max(1e-4,(args.schedule_step-step)/(args.schedule_step-args.warmup_step))

def lr_schedule(init_lr, step, args):
    if args.warmup:
        return warmup_linear(args, step)*init_lr
    else:
        return init_lr

def init_world_size(world_size):
    assert world_size <= torch.cuda.device_count()
    return torch.cuda.device_count() if world_size == -1 else world_size

def check_args_environment(args):
    if not torch.cuda.is_available():
        logging.warning("Cuda is not available, " \
                        "related options will be disabled")
    args.enable_gpu = torch.cuda.is_available() & args.enable_gpu
    return args


class timer:
    """
    Time context manager for code block
    """
    from time import time
    NAMED_TAPE = {}

    def __init__(self, name, **kwargs):
        self.name = name
        if name not in timer.NAMED_TAPE:
            timer.NAMED_TAPE[name] = 0

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        timer.NAMED_TAPE[self.name] += timer.time() - self.start
        print(self.name, timer.NAMED_TAPE[self.name])


def load_glove(path, word_dict, word_count_map):
    print("Loading Glove Model")
    glove_embs = []
    glove_words = []
    
    with open(path,'r') as f:
        for line in f:
            split_line = line.split()
            
            word = split_line[0]
            if word in word_dict:
                if len(split_line) == 301:
                    embedding = np.array(split_line[1:], dtype=np.float64)
                    word_dict[word] = len(word_dict) + 2
                    glove_words.append(word)
                    glove_embs.append(embedding)
                    word_count_map[word] = 1
                    
    print(f"{len(glove_words)} words loaded!")
    print(len(glove_embs))
    glove_embs = torch.as_tensor(glove_embs, dtype=torch.float)
    return glove_words, glove_embs, word_count_map


def load_file(path, label_map):
    all_text = []
    all_label = []
    with open(path, 'r') as f:
        for line in f:
            news_info = line.strip('\n').split('\t')
            label_name = news_info[1]
            if label_name not in label_map:
                label_map[label_name] = len(label_map)
            text = " ".join([news_info[1], news_info[2], news_info[3]])
            all_text.append(text)
            all_label.append(label_map[label_name])
    return all_text, all_label, label_map


def load_mind(root):
    train_path = os.path.join(root, 'MINDlarge_train/news.tsv')
    test_path = os.path.join(root, 'MINDlarge_test/news.tsv')
    dev_path = os.path.join(root, 'MINDlarge_dev/news.tsv')

    label_map = {}

    dataset = {'train': {'text': [], 'label': []},
               'test': {'text': [], 'label': []},
               'dev': {'text': [], 'label': []}}

    all_text, all_label, label_map = load_file(train_path, label_map)
    dataset['train']['text'] = all_text
    dataset['train']['label'] = all_label

    all_text, all_label, label_map = load_file(test_path, label_map)
    dataset['test']['text'] = all_text
    dataset['test']['label'] = all_label

    all_text, all_label, label_map = load_file(dev_path, label_map)
    dataset['dev']['text'] = all_text
    dataset['dev']['label'] = all_label

    dataset['class_num'] = len(label_map)
    return dataset      

def prepare_glove():
    dataset = load_mind('/workspaceblobstore/v-wenjunpeng/mind/')
    text=[]
    label=[]
    
    for row in dataset['train']['text']+dataset['test']['text']+dataset['dev']['text']:
        tokens = wordpunct_tokenize(row.lower())
        text.append(tokens)
    word_dict= {'[PAD]':0, '[UNK]':1}
    word_count_map = {'[PAD]':1, '[UNK]':1}
    for sent in text:    
        for token in sent:        
            if token not in word_dict:
                word_dict[token]=len(word_dict)
                word_count_map[token] = 0
    print(len(word_dict))
    glove_words, glove_embs, word_count_map = load_glove('/workspaceblobstore/v-wenjunpeng/glove/glove.840B.300d.txt', word_dict, word_count_map)
    word_count_map['[PAD]'] = 1
    word_count_map['[UNK]'] = 1
    torch.save(glove_embs, '../data/glove/glove_embs_for_mind.pt')

    with open('../data/glove/vocab.txt', 'w') as f:
        f.write('[PAD]\n')
        f.write('[UNK]\n')
        for w in glove_words:
            f.write(w+'\n')

    with open('../data/glove/count_map.json', 'w') as f:
        f.write(json.dumps(word_count_map))
    
    
if __name__ == '__main__':
    if not os.path.exists('../data/glove'):
        os.mkdir('../data/glove')
    prepare_glove()
    with open('../data/glove/count_map.json', 'r') as f:
        wcm = json.loads(f.read())
        print(len(wcm))

