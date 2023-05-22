import csv
import os.path
import random

import pandas as pd


def save_data(save_dir, headers, data, file_name):
    save_file_path = save_dir + file_name + '.csv'
    with open(save_file_path, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        writer.writerow(headers)
        writer.writerows(data)


def split(file_path, save_dir, key_title):
    if not os.path.exists(file_path):
        return
    full_data = []
    key_photo_id = 'photo_id'
    key_summary = 'summary'
    train_ratio, dev_ratio = 0.8, 0.1
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        keys = df.keys()
        print('ZFC df keys = {}'.format(keys))
        df = df[[key_photo_id, key_title, key_summary]]
        for index, row in df.iterrows():
            photo_id = row[key_photo_id]
            title = row[key_title]
            summary = row[key_summary]
            full_data.append((photo_id, title, summary))
            print('ZFC split index = {}, photo_id = {}, title = {}, summary = {}'.format(index, photo_id, title, summary))
    full_len = len(full_data)
    random.shuffle(full_data)
    train_len = int(full_len * train_ratio)
    train_data = full_data[: train_len]
    dev_len = int(full_len * dev_ratio) + train_len
    dev_data = full_data[train_len: dev_len]
    test_data = full_data[dev_len:]
    all_data = [(train_data, 'train'), (dev_data, 'dev'), (test_data, 'test')]
    headers = [key_photo_id, key_title, key_summary]
    for (data, file_name) in all_data:
        save_data(save_dir, headers, data, file_name)


if __name__ == '__main__':
    data_file_path = '../resources/t5_pegasus_dataset.csv'
    save_path = '../dataset/valid_title/'
    split(data_file_path, save_path, 'valid_title')
    data_file_path = '../resources/t5_pegasus_dataset_pure.csv'
    save_path = '../dataset/pure_title/'
    split(data_file_path, save_path, 'pure_title')
