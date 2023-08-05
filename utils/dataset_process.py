import csv
import os.path
import random

import pandas as pd

import utils

OCR_START = '[OS]'
OCR_END = '[OE]'
MMU_OCR_START = '[MOS]'
MMU_OCR_END = '[MOE]'
IMAGE_CAPTION_START = '[ICS]'
IMAGE_CAPTION_END = '[ICE]'

SPLIT_SYMBOL = '，'


def save_data(save_dir, headers, data, file_name):
    save_file_path = save_dir + utils.FILE_SPLIT_SYMBOL + file_name + '.csv'
    with open(save_file_path, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        writer.writerow(headers)
        writer.writerows(data)


def rectify_summary_by_max_len(summary, max_len):
    summary_len = len(summary)
    if summary_len <= max_len:
        return summary
    os_index = summary.find(OCR_START)
    oe_index = summary.find(OCR_END)
    mos_index = summary.find(MMU_OCR_START)
    moe_index = summary.find(MMU_OCR_END)
    ics_index = summary.find(IMAGE_CAPTION_START)
    ice_index = summary.find(IMAGE_CAPTION_END)
    se_pairs = [
        (mos_index, moe_index, MMU_OCR_START, MMU_OCR_END),
        (os_index, oe_index, OCR_START, OCR_END),
        (ics_index, ice_index, IMAGE_CAPTION_START, IMAGE_CAPTION_END)
    ]
    valid_summary = ''
    for (s_index, e_index, s_symbol, e_symbol) in se_pairs:
        if s_index < -1:  # 当前 Start symbol 包裹的字符串不存在，代表整个 symbol 句子不存在 --> 终止循环
            continue
        valid_summary_len = len(valid_summary)
        # 当前摘要长度已经达到 max_len --> 终止循环
        if valid_summary_len >= max_len:
            break
        e_symbol_len = len(e_symbol)
        text = summary[s_index: e_index + e_symbol_len]  # 取出 symbol 及其包裹的内容
        valid_text = text.replace(s_symbol, '').replace(e_symbol, '')  # 过滤掉 symbol
        valid_text_splits = valid_text.split(SPLIT_SYMBOL)  # 按分隔符将内容分割为句子列表
        first_text = valid_text_splits[0]  # 取出第一个句子

        # 如果当前剩余空间已经无法放得当前这对symbol以及该symbol内容中的首个句子 --> 终止循环
        if max_len - valid_summary_len - len(s_symbol) - e_symbol_len - len(first_text) < 0:
            break
        valid_summary += (s_symbol + first_text)
        for text_split in valid_text_splits[1:]:
            valid_summary_len = len(valid_summary)
            text_split_len = len(text_split) + len(SPLIT_SYMBOL)  # + len(SPLIT_SYMBOL) 是因为之前分割时被删除的分隔符要加回来
            # 如果当前剩余空间减去 End symbol 后已经无法放得下当前句子及一个分隔符 --> 终止循环
            if max_len - valid_summary_len - e_symbol_len - text_split_len < 0:
                break
            valid_summary += (SPLIT_SYMBOL + text_split)
        valid_summary += e_symbol
    return valid_summary


def split(file_path, key_title, max_len, name='short_video'):
    save_dir = '../dataset/{}'.format(name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(file_path):
        return
    full_data = []
    key_photo_id = utils.KEY_PHOTO_ID
    key_summary = utils.KEY_SUMMARY
    key_is_mmu = 'is_mmu'
    key_ground_truth = utils.KEY_GROUND_TRUTH
    train_ratio, dev_ratio = 0.8, 0.1
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    keys = df.keys()
    print('ZFC df keys = {}'.format(keys))
    is_msrvtt = name == 'msrvtt'
    if is_msrvtt:
        df = df[[key_photo_id, key_is_mmu, key_title, key_summary, key_ground_truth]]
    else:
        df = df[[key_photo_id, key_is_mmu, key_title, key_summary]]
    for index, row in df.iterrows():
        photo_id = row[key_photo_id]
        title = row[key_title]
        summary = row[key_summary]
        summary = rectify_summary_by_max_len(summary, max_len)
        is_mmu = row[key_is_mmu]
        if is_msrvtt:
            ground_truth = row[key_ground_truth]
            full_data.append((photo_id, is_mmu, title, summary, ground_truth))
        else:
            full_data.append((photo_id, is_mmu, title, summary))
        print('ZFC split index = {}, photo_id = {}, title = {}, summary = {}'.format(index, photo_id, title, summary))
    full_len = len(full_data)
    random.shuffle(full_data)
    train_len = int(full_len * train_ratio)
    train_data = full_data[: train_len]
    dev_len = int(full_len * dev_ratio) + train_len
    dev_data = full_data[train_len: dev_len]
    test_data = full_data[dev_len:]
    all_data = [(train_data, 'train'), (dev_data, 'dev'), (test_data, 'test')]
    headers = [key_photo_id, key_is_mmu, key_title, key_summary]
    if is_msrvtt:
        headers.append(key_ground_truth)
    for (data, file_name) in all_data:
        save_data(save_dir, headers, data, file_name)


def stat_summary_max_len(key_title, max_len):
    data_dir = '../dataset/{}/{}'.format(key_title, max_len)
    csv_list = os.listdir(data_dir)
    curr_max_len = 0
    for csv_file in csv_list:
        file_path = data_dir + '/' + csv_file
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        df = df[utils.KEY_SUMMARY]
        for index, item in df.items():
            curr_max_len = max(curr_max_len, len(item))
            if curr_max_len > max_len:
                print('ZFC stat_summary_max_len beyond max_len = {}, curr_max_len = {}, file_path = {}, index = {}, \
                summary = {}'.format(max_len, curr_max_len, file_path, index, item))
    print('ZFC stat_summary_max_len final curr_max_len = {}, data_dir = {}'.format(curr_max_len, data_dir))


if __name__ == '__main__':
    params = [
        # ('../resources/dataset.csv', utils.KEY_TITLE, 1536, 'short_video')
        ('../resources/msrvtt_dataset.csv', utils.KEY_TITLE, 1536, 'msrvtt')
    ]
    for (file_path, key_title, max_len, name) in params:
        split(file_path, key_title, max_len, name)
        # stat_summary_max_len(key_title, max_len)
