import json
import os
import re
import time

import rouge
import jieba
import torch
import argparse
import numpy as np
import pandas as pd
from torch import optim
from tqdm.auto import tqdm
from bert4torch.models import *
from torch.utils.data import DataLoader, Dataset

from eval.score import calc_scores
from utils import utils

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs

from torch._six import string_classes
int_classes = int
from transformers import MT5ForConditionalGeneration, BertTokenizer, T5ForConditionalGeneration, AutoTokenizer, \
    AutoModelForSeq2SeqLM


def load_data(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f.readlines():
            cur = l.strip().split('\t')
            if len(cur) == 2:
                title, content = cur[0], cur[1]
                D.append((title, content))
            elif len(cur) == 1:
                content = cur[0]
                D.append(content)
    return D


class T5PegasusTokenizer(BertTokenizer):
    """结合中文特点完善的Tokenizer
    基于词颗粒度的分词，如词表中未出现，再调用BERT原生Tokenizer
    """
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


class KeyDataset(Dataset):
    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def create_data(data, tokenizer, max_len=512, term='train'):
    """调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
    """
    start = time.time_ns() / 1000000
    ret, flag = [], True
    for title, content in tqdm(data):
        text_ids = tokenizer.encode(content, max_length=max_len, truncation='only_first')
        if flag and term == 'train':
            flag = False
            print(content)
        if term == 'train':
            summary_ids = tokenizer.encode(title, max_length=max_len, truncation='only_first')
            features = {'input_ids': text_ids,
                        'decoder_input_ids': summary_ids,
                        'attention_mask': [1] * len(text_ids),
                        'decoder_attention_mask': [1] * len(summary_ids)
                       }
        
        elif term == 'dev':
            features = {'input_ids': text_ids,
                        'attention_mask': [1] * len(text_ids),
                        'title': title
                       }
            
        ret.append(features)
    spend = time.time_ns() / 1000000 - start
    print('ZFC create_data term = {}, spend_time = {}'.format(term, spend))
    return ret


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')


def default_collate(batch):
    """组batch
    各个数据域分别转换为tensor，tensor第一个维度等于batch_size
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            batch = sequence_padding(batch)
        return default_collate([default_collate(elem) for elem in batch])
    raise TypeError(default_collate_err_msg_format.format(elem_type))


def prepare_data(args, data_path, tokenizer, term='train', data_type=''):
    """准备batch数据
    """
    if utils.is_short_video_dataset(data_type):
        data = utils.load_short_video_data(data_path, data_type)
    else:
        data = load_data(data_path)
    data = create_data(data, tokenizer, args.max_len, term)
    data = KeyDataset(data)
    data = DataLoader(data, batch_size=args.batch_size, collate_fn=default_collate)
    return data


def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l
    """
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.Rouge().get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }
    
    
def compute_rouges(sources, targets):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]

    return {k: v / len(targets) for k, v in scores.items()}


def train_model(model, optimizer, train_data, dev_data, tokenizer, device, args):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
        
    best_rouge_l = 0
    best_cider = 0
    total_loss = {}
    epoch_loss = []
    total_epoch = int(args.num_epoch)
    save_epoch_interval = int(args.save_epoch_interval)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch, eta_min=0, last_epoch=-1)
    for epoch in range(total_epoch):
        model.train()  # 指明当前是训练阶段
        accu_train_loss = []
        mini_train_loss = []
        optimizer_lrs = []
        scheduler_lrs = []
        for i, cur in enumerate(tqdm(train_data, desc='Epoch {}:'.format(epoch))):
            cur = {k: v.to(device) for k, v in cur.items()}
            prob = model(**cur)[0]  # 计算当前样本的结果
            mask = cur['decoder_attention_mask']
            mask = mask[:, 1:]
            mask = mask.reshape(-1)
            mask = mask.bool()
            prob = prob[:, :-1]
            prob = prob.reshape((-1, prob.size(-1)))
            prob = prob[mask]
            labels = cur['decoder_input_ids']
            labels = labels[:, 1:]
            labels = labels.reshape(-1)
            labels = labels[mask]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prob, labels)  # 根据当前样本的计算结果 & 标签 --> 计算Loss
            accu_train_loss.append(loss.detach().cpu().item())
            loss_value = loss.detach().cpu().item()
            if loss_value <= 10:
                mini_train_loss.append(loss_value)
            else:
                mini_train_loss.append(10)
            if i % 100 == 0:
                print("Iter {}:  Training Loss: {}".format(i, loss.item()))
                utils.draw_train_loss_curve(mini_train_loss, epoch)
                utils.draw_optimizer_lr_curve(optimizer_lrs, epoch)
                utils.draw_scheduler_lr_curve(scheduler_lrs, epoch)
            optimizer_lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
            scheduler_lrs.append(lr_scheduler.get_last_lr())
            loss.backward()  # 开启后向传播
            optimizer.step()  # 更新模型参数
            optimizer.zero_grad()  # 梯度清零
        accu_train_loss = accu_train_loss
        avg_train_loss = sum(accu_train_loss) / len(accu_train_loss)
        epoch_loss.append(avg_train_loss)

        lr_scheduler.step()

        # 验证
        model.eval()  # 指明当前是验证阶段
        gens = []
        summaries = []
        for feature in tqdm(dev_data):
            title = feature['title']
            content = {k : v.to(device) for k, v in feature.items() if k != 'title'} 
            if args.data_parallel and torch.cuda.is_available():
                gen = model.module.generate(max_length=args.max_len_generate,
                             eos_token_id=tokenizer.sep_token_id,
                             decoder_start_token_id=tokenizer.cls_token_id,
                             **content)
            else:
                gen = model.generate(max_length=args.max_len_generate,
                             eos_token_id=tokenizer.sep_token_id,
                             decoder_start_token_id=tokenizer.cls_token_id,
                             **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [item.replace(' ', '') for item in gen]
            # print(title)
            # print(gen)
            gens.extend(gen)
            summaries.extend(title)
        scores = calc_scores(gens, summaries)
        utils.draw_score_curve(scores)
        cider = scores['CIDEr'].item()
        print("Validation Loss: {}".format(scores))
        rouge_l = scores['rouge-l']
        model_name = 'best_summary_model'
        if rouge_l > best_rouge_l:
            best_rouge_l = rouge_l
            save_name = model_name + '_rouge_l'
            if args.data_parallel and torch.cuda.is_available():
                torch.save(model.module, os.path.join(args.model_dir, save_name))
            else:
                torch.save(model, os.path.join(args.model_dir, save_name))
            total_loss['best_rouge_l'] = {
                'avg_train_loss': avg_train_loss,
                'accu_train_loss': accu_train_loss,
                'scores': scores,
                'rouge_l': rouge_l,
                'epoch': epoch
            }

        if cider > best_cider:
            best_cider = cider
            save_name = model_name + '_cider'
            if args.data_parallel and torch.cuda.is_available():
                torch.save(model.module, os.path.join(args.model_dir, save_name))
            else:
                torch.save(model, os.path.join(args.model_dir, save_name))
            total_loss['best_cider'] = {
                'avg_train_loss': avg_train_loss,
                'accu_train_loss': accu_train_loss,
                'scores': scores,
                'cider': cider,
                'epoch': epoch
            }

        total_loss[epoch] = {
            'avg_train_loss': avg_train_loss,
            'accu_train_loss': accu_train_loss,
            'scores': scores,
            'rouge_l': rouge_l,
            'cider': cider
        }
        epoch_plus = epoch + 1
        if epoch_plus % save_epoch_interval == 0 or epoch_plus == total_epoch:
            utils.check_mkdirs('./checkpoint')
            torch.save(model, './checkpoint/{}_{}'.format(model_name, str(epoch)))
            utils.send_wechat_msg('train_with_finetune save_model', 'epoch = {}, rouge_l = {}, cider = {}'
                                  .format(epoch, rouge_l, cider))

        if epoch % 4 == 0:
            utils.send_wechat_msg('train_with_finetune', 'epoch = {}, rouge_l = {}, cider = {}'
                                  .format(epoch, rouge_l, cider))
    utils.draw_epoch_loss_curve(epoch_loss)
    utils.save_msg_to_local(json.dumps(total_loss), 'TrainLossScore.txt')
    utils.send_wechat_msg('train_with_finetune', 'train_model finished')
    utils.check_shutdown()


DATA_TYPE = utils.DATA_TYPE_PURE


def init_argument():
    bz, max_len = utils.get_bz_max_len(DATA_TYPE)
    parser = argparse.ArgumentParser(description='t5-pegasus-chinese')
    parser.add_argument('--train_data', default='./dataset/short_video/train.csv')
    parser.add_argument('--dev_data', default='./dataset/short_video/dev.csv')
    parser.add_argument('--pretrain_model', default='./t5_pegasus_pretrain')
    parser.add_argument('--model_dir', default='./saved_model')
    
    parser.add_argument('--num_epoch', default=20, help='number of epoch')
    parser.add_argument('--batch_size', default=bz, help='batch size')
    parser.add_argument('--lr', default=4e-5, help='learning rate')
    parser.add_argument('--data_parallel', default=False)
    parser.add_argument('--max_len', default=max_len, help='max length of inputs')
    parser.add_argument('--max_len_generate', default=64, help='max length of outputs')
    parser.add_argument('--data_type', default=DATA_TYPE)
    parser.add_argument('--model_type', default='mt5')
    parser.add_argument('--save_epoch_interval', type=int, default=9)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    try:
        # step 1. init argument
        args = init_argument()
        is_mt5 = args.model_type == 'mt5'

        # step 2. prepare training data and validation data
        if is_mt5:
            tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)
        special_tokens_dict = {'additional_special_tokens': ['[OS]', '[OE]', '[MOS]', '[MOE]', '[ICS]', '[ICE]']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

        train_data = prepare_data(args, args.train_data, tokenizer, term='train', data_type=DATA_TYPE)
        dev_data = prepare_data(args, args.dev_data, tokenizer, term='dev', data_type=DATA_TYPE)

        # step 3. load pretrain model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if is_mt5:
            model = MT5ForConditionalGeneration.from_pretrained(args.pretrain_model).to(device)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrain_model).to(device)
        model.resize_token_embeddings(len(tokenizer))
        if args.data_parallel and torch.cuda.is_available():
            device_ids = range(torch.cuda.device_count())
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        # step 4. finetune
        adam = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_model(model, adam, train_data, dev_data, tokenizer, device, args)
    except Exception as e:
        name = 'train_with_finetune'
        msg = 'exception: {}'.format(str(e))
        print(name + ' ' + msg)
        utils.save_msg_to_local(name + ' ' + msg, 'TrainFinetuneException.txt')
        utils.send_wechat_msg(name, msg)
        utils.check_shutdown()

