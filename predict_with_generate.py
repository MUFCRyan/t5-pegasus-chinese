import math
import time

from tqdm import tqdm
from transformers import MT5ForConditionalGeneration, AutoTokenizer
import jieba
from transformers import BertTokenizer, BatchEncoding
import torch

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
from torch.utils.data import DataLoader, Dataset
import re
import os
import csv
import argparse
from multiprocessing import Pool, Process
import pandas as pd
import numpy as np
import rouge


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(filename):
    """加载数据
    单条格式：(正文) 或 (标题, 正文)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def pre_tokenizer(self, x):
        return jieba.cut(x, HMM=False)

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
    
    
def create_data(data, tokenizer, max_len, is_mt5=True):
    """调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
    """
    ret, flag, title, photo_id = [], True, None, None
    for content in data:
        ground_truth = None
        if type(content) == tuple:
            photo_id, title, content, *ground_truth = content
        text_ids = tokenizer.encode(content, max_length=max_len,
                                    truncation='only_first')

        if flag:
            flag = False
            print(content)

        features = {'input_ids': text_ids,
                    'attention_mask': [1] * len(text_ids),
                    'raw_data': content}
        if title:
            features['title'] = title
        if photo_id:
            features[utils.KEY_PHOTO_ID] = str(photo_id)
        if features is not None and ground_truth is not None and len(ground_truth) > 0 and not is_mt5:
            features[utils.KEY_GROUND_TRUTH] = ground_truth[0]
        ret.append(features)
    return ret


def create_extract_data(data, tokenizer, max_len=512, term='train'):
    """调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
    """
    start = time.time_ns() / 1000000
    ret, flag = [], True
    for pid, title, content in tqdm(data):
        if type(title) is float and math.isnan(title):
            title = ''
        text_ids = tokenizer.encode(content, max_length=max_len, truncation='only_first')
        if flag and term == 'train':
            flag = False
            print(content)
        if term == 'train':
            summary_ids = tokenizer.encode(title, max_length=max_len, truncation='only_first')
            features = {'input_ids': text_ids,
                        'decoder_input_ids': summary_ids,
                        'attention_mask': [1] * len(text_ids),
                        'decoder_attention_mask': [1] * len(summary_ids),
                        utils.KEY_PHOTO_ID: str(pid)
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
        return torch.stack(batch, 0, out=out).to(device)
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
        datas = {}
        for key in elem:
            if key in [utils.KEY_GROUND_TRUTH]:
                datas[key] = [d[key] for d in batch]
            else:
                datas[key] = default_collate([d[key] for d in batch])
        return datas
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
    

def prepare_data(args, tokenizer, data_type='', is_extract=False, use_gt=False):
    """准备batch数据
    """
    if is_extract:
        data_path = utils.PREDICT_DIR + '_' + args.model_type + utils.FILE_SPLIT_SYMBOL + args.mode + utils.SUFFIX_CSV
    else:
        data_path = args.test_data

    if utils.is_short_video_dataset(data_type):
        test_data = utils.load_short_video_data(data_path, data_type, True, need_pid=True, use_gt=use_gt)
    else:
        test_data = load_data(data_path)

    if is_extract:
        test_data = create_extract_data(test_data, tokenizer, int(args.max_len))
    else:
        test_data = create_data(test_data, tokenizer, int(args.max_len), is_mt5=False)

    test_data = KeyDataset(test_data)
    test_data = DataLoader(test_data, batch_size=int(args.batch_size), collate_fn=default_collate, shuffle=False)
    return test_data


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


def generate_summary(test_data, model, tokenizer, args, is_mt5=True, use_gt=False):
    gens, summaries = [], []
    mode = args.mode
    predict_dir = utils.PREDICT_DIR + '_' + args.model_type
    utils.check_mkdirs(predict_dir)
    predict_file = predict_dir + utils.FILE_SPLIT_SYMBOL + mode + utils.SUFFIX_CSV
    has_ground_truth = False
    with open(predict_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow((utils.KEY_PHOTO_ID, 'real_title', utils.KEY_TITLE, utils.KEY_SUMMARY))
        model.eval()
        ground_truthes = []
        for feature in tqdm(test_data):
            if not has_ground_truth and utils.KEY_GROUND_TRUTH in feature.keys():
                has_ground_truth = True
            if has_ground_truth:
                ground_truth = feature[utils.KEY_GROUND_TRUTH]
            else:
                ground_truth = None
            real_title = feature['title']
            raw_data = feature['raw_data']
            photo_ids = feature[utils.KEY_PHOTO_ID]
            photo_ids = {int(pid) if is_mt5 else pid for pid in photo_ids}
            content = {k: v for k, v in feature.items() if k not in ['raw_data', 'title', utils.KEY_PHOTO_ID, utils.KEY_GROUND_TRUTH]}
            gen = model.generate(max_length=args.max_len_generate,
                                 eos_token_id=tokenizer.sep_token_id,
                                 decoder_start_token_id=tokenizer.cls_token_id,
                                 **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            if is_mt5:
                gen = [item.replace(' ', '') for item in gen]
            writer.writerows(zip(photo_ids, real_title, gen, raw_data))
            gens.extend(gen)
            if 'title' in feature:
                summaries.extend(feature['title'])
            if has_ground_truth and ground_truth is not None:
                ground_truthes.extend(ground_truth)
    if len(summaries) > 0:
        scores = calc_scores(gens, summaries, is_mt5)
        print('scores = {}'.format(scores))
        if has_ground_truth and len(ground_truthes) > 0:
            scores = calc_scores(gens, ground_truthes, is_mt5, is_ground_truth=True)
            print('ground_truth scores = {}'.format(scores))
    print('Done!')


def extract(test_data, model, mode, feat_type, is_mt5=True, num_frames=-1):
    with torch.no_grad():
        model.eval()
        if is_mt5:
            root_dir = utils.SAVE_PATH_FEATURES
        else:
            root_dir = utils.SAVE_PATH_MSRVTT_FEATURES
        save_dir = root_dir + utils.FILE_SPLIT_SYMBOL + feat_type
        if num_frames > 0:
            save_dir += ('_' + str(num_frames))
        utils.check_mkdirs(save_dir)
        for index, (photo_id, feature) in enumerate(tqdm(test_data)):
            content = {k: v.to(device) for k, v in feature.items()}
            result = model(**content)
            last_hidden_state = result.encoder_last_hidden_state.cpu().squeeze(0)
            hidden_state = last_hidden_state.half()
            feature_list = [(photo_id, hidden_state)]
            utils.save_features(save_dir, mode, feature_list, False)


def generate_multiprocess(feature):
    """多进程
    """
    model.eval()
    raw_data = feature['raw_data']
    content = {k: v for k, v in feature.items() if k != 'raw_data'}
    gen = model.generate(max_length=args.max_len_generate,
                             eos_token_id=tokenizer.sep_token_id,
                             decoder_start_token_id=tokenizer.cls_token_id,
                             **content)
    gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
    results = ["{}\t{}".format(x.replace(' ', ''), y) for x, y in zip(gen, raw_data)]
    return results


def init_argument():
    parser = argparse.ArgumentParser(description='t5-pegasus-chinese')
    parser.add_argument('--test_data', default='./data/predict.tsv')
    parser.add_argument('--result_file', default='./data/predict_result.csv')
    parser.add_argument('--pretrain_model', default='./t5_pegasus_pretrain')    
    parser.add_argument('--model', default='./saved_model/remote_summary_model')

    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--max_len', default=512, help='max length of inputs')
    parser.add_argument('--max_len_generate', default=64, help='max length of generated text')
    parser.add_argument('--use_multiprocess', default=False, action='store_true')
    parser.add_argument('--extract', type=str, default='False', help='if use extract text features')
    parser.add_argument('--use_gt', type=str, default='False')
    parser.add_argument('--mode', default='temp')
    parser.add_argument('--model_type', default='mt5')
    parser.add_argument('--num_frames', type=str, default=-1)

    args = parser.parse_args()
    return args


def filter_data(test_data, feat_type, is_mt5=True):
    data_list = []
    for feature in tqdm(test_data, desc='filter_data, feat_type = {}'.format(feat_type)):
        photo_id_list = feature[utils.KEY_PHOTO_ID]
        input_id_list = torch.unbind(feature['input_ids'], dim=0)
        attention_mask_list = torch.unbind(feature['attention_mask'], dim=0)
        decoder_input_id_list = torch.unbind(feature['decoder_input_ids'], dim=0)
        decoder_attention_mask_list = torch.unbind(feature['decoder_attention_mask'], dim=0)
        data_len = len(input_id_list)
        for index in range(data_len):
            photo_id = photo_id_list[index]
            if feat_type == utils.FEATURE_TYPE_CONTENT:
                input_ids = input_id_list[index]
                attention_mask = attention_mask_list[index]
                decoder_input_ids = decoder_input_id_list[index]
                decoder_attention_mask = decoder_attention_mask_list[index]
            else:
                input_ids = decoder_input_id_list[index]
                attention_mask = decoder_attention_mask_list[index]
                decoder_input_ids = input_id_list[index]
                decoder_attention_mask = attention_mask_list[index]
            data_list.append((photo_id, {
                'input_ids': input_ids.unsqueeze(0),
                'decoder_input_ids': decoder_input_ids.unsqueeze(0),
                'attention_mask': attention_mask.unsqueeze(0),
                'decoder_attention_mask': decoder_attention_mask.unsqueeze(0)
            }))
    return data_list



if __name__ == '__main__':
    
    # step 1. init argument
    args = init_argument()
    is_mt5 = args.model_type == 'mt5'
    is_extract = args.extract == str(True)
    use_gt = args.use_gt == str(True)

    num_frames = int(args.num_frames)

    # step 2. prepare test data
    if is_mt5:
        tokenizer_config_path = './t5_pegasus_pretrain'
    else:
        tokenizer_config_path = 'google/flan-t5-base'
    if is_mt5:
        tokenizer = T5PegasusTokenizer.from_pretrained(tokenizer_config_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config_path)
    special_tokens_dict = {'additional_special_tokens': ['[OS]', '[OE]', '[MOS]', '[MOE]', '[ICS]', '[ICE]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    test_data = prepare_data(args, tokenizer, utils.DATA_TYPE_PURE, is_extract=is_extract, use_gt=use_gt)
    
    # step 3. load finetuned model
    model = torch.load(args.model, map_location=device)
    model.resize_token_embeddings(len(tokenizer))

    if is_extract:
        curr_data = filter_data(test_data, utils.FEATURE_TYPE_SUMMARY, is_mt5)
        extract(curr_data, model, args.mode, utils.FEATURE_TYPE_SUMMARY, is_mt5, num_frames)
        curr_data = filter_data(test_data, utils.FEATURE_TYPE_CONTENT, is_mt5)
        extract(curr_data, model, args.mode, utils.FEATURE_TYPE_CONTENT, is_mt5, num_frames)
    else:
        # step 4. predict
        res = []
        if args.use_multiprocess and device == 'cpu':
            print('Parent process %s.' % os.getpid())
            p = Pool(2)
            res = p.map_async(generate_multiprocess, test_data, chunksize=2).get()
            print('Waiting for all subprocesses done...')
            p.close()
            p.join()
            res = pd.DataFrame([item for batch in res for item in batch])
            res.to_csv(args.result_file, index=False, header=False, encoding='utf-8')
            print('Done!')
        else:
            generate_summary(test_data, model, tokenizer, args, is_mt5=is_mt5, use_gt=use_gt)