import os
import pickle
import platform

import pandas as pd
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm

KEY_IS_MMU = 'is_mmu'
KEY_PHOTO_ID = "photo_id"
_ROOT_DIR = 'E:'
if platform.system().lower() == 'linux':
    _ROOT_DIR = ''
SAVE_PATH_FEATURES = _ROOT_DIR + '/Dataset/NLP/ShortVideo/Kuaishou/features/text'

FILE_SPLIT_SYMBOL = '/'


PREDICT_DIR = './data/predict'


KEY_VALID_TITLE = 'valid_title'
KEY_TITLE = 'title'
KEY_PHOTO_ID = 'photo_id'
KEY_SUMMARY = 'summary'


SUFFIX_PKL = ".pkl"
SUFFIX_CSV = ".csv"


DATA_TYPE_VALID = KEY_VALID_TITLE
DATA_TYPE_PURE = KEY_TITLE


FEATURE_TYPE_SUMMARY = 'summary'
FEATURE_TYPE_CONTENT = 'content'


def check_mkdirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_train_path(data_type='', max_len=1024):
    if data_type in [DATA_TYPE_VALID, DATA_TYPE_PURE]:
        return './dataset/{}/{}/train.csv'.format(data_type, max_len)
    else:
        return './data/train.tsv'


def get_dev_path(data_type='', max_len=1024):
    if data_type in [DATA_TYPE_VALID, DATA_TYPE_PURE]:
        return './dataset/{}/{}/dev.csv'.format(data_type, max_len)
    else:
        return './data/dev.tsv'


def get_test_path(data_type='', max_len=1024):
    if data_type in [DATA_TYPE_VALID, DATA_TYPE_PURE]:
        return './dataset/{}/{}/test.csv'.format(data_type, max_len)
    else:
        return './data/test.tsv'


def get_key_title(data_type=''):
    if data_type == DATA_TYPE_VALID:
        return KEY_VALID_TITLE
    else:
        return KEY_TITLE


def is_short_video_dataset(data_type=''):
    return data_type in [DATA_TYPE_VALID, DATA_TYPE_PURE]


def get_bz_max_len(data_type=''):
    if is_short_video_dataset(data_type):
        return 1, 1536
    else:
        return 16, 512


def load_short_video_data(file_name, data_type, need_title=True, need_pid=False):
    data = []
    key_photo_id = KEY_PHOTO_ID
    key_title = get_key_title(data_type)
    key_summary = KEY_SUMMARY
    df = pd.read_csv(file_name, sep='\t', encoding='utf-8')
    df = df[[key_photo_id, key_title, key_summary]]
    for index, row in df.iterrows():
        photo_id = row[key_photo_id]
        summary = row[key_summary]
        if need_title:
            title = row[key_title]
            if need_pid:
                data.append((photo_id, title, summary))
            else:
                data.append((title, summary))
        else:
            if need_pid:
                data.append((photo_id, summary))
            else:
                data.append(summary)
    return data


def draw_loss_curve(epoch_losses):
    plt.switch_backend('Agg')  # 后端设置'Agg' 参考：https://cloud.tencent.com/developer/article/1559466
    plt.figure()  # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
    plt.plot(epoch_losses, 'b', label='loss')  # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()  # 个性化图例（颜色、形状等）
    plt.savefig("./1_recon_loss.jpg")  # 保存图片 路径：/imgPath/


def save_msg_to_local(msg, file_path):
    mode = 'w+'
    if os.path.exists(file_path):
        mode = 'r+'
    with open(file_path, mode) as f:
        f.write(msg)


def is_linux():
    return platform.system().lower() == 'linux'


def save_by_pickle(root_dir, photo_id, feature, suffix):
    file_path = root_dir + FILE_SPLIT_SYMBOL + str(photo_id) + suffix
    pickle.dump(feature, file=open(file_path, 'wb+'))


def save_features(save_dir, data_type, feature_list, print_info=True):
    if not data_type.startswith(FILE_SPLIT_SYMBOL):
        data_type = FILE_SPLIT_SYMBOL + data_type
    save_dir += data_type
    check_mkdirs(save_dir)
    for (photo_id, feature) in tqdm(feature_list):
        save_by_pickle(save_dir, photo_id, feature, SUFFIX_PKL)
    if print_info:
        print('ZFC save_features save_dir = {} succeed!'.format(save_dir))


def check_shutdown():
    if is_linux():
        os.system("/usr/bin/shutdown")


_TOKEN = 'eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjE0MzAyMSwidXVpZCI6IjgyNTY3ZjBmLWJmYzUtNDhhNS1iNGUxLWMzNGEzODlmMTAwOCIsImlzX2FkbWluIjpmYWxzZSwiaXNfc3VwZXJfYWRtaW4iOmZhbHNlLCJzdWJfbmFtZSI6IiIsInRlbmFudCI6ImF1dG9kbCIsInVwayI6IiJ9.qYYNIo8gkliLAsssn-CW5Qwors91mQTrP4-nrWHrzxBT7JVifhuKKP9C_ZnbPQqDfACnpBsjybHjmbmF-YkIkg'
headers = {"Authorization": _TOKEN}


def send_wechat_msg(name, msg):
    if not is_linux():
        return
    text = name + ' ' + msg
    resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                         json={
                             "title": "t5-pegasus-chinese",
                             "name": text
                         }, headers=headers)
    print(resp.content.decode())


# send_wechat_msg('Test', '测试微信消息发送')
