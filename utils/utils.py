KEY_VALID_TITLE = 'valid_title'
KEY_PURE_TITLE = 'pure_title'
KEY_SUMMARY = 'summary'


DATA_TYPE_VALID = 'valid'
DATA_TYPE_PURE = 'pure'


def get_train_path(data_type=''):
    if data_type == DATA_TYPE_VALID:
        return './dataset/valid_title/train.csv'
    if data_type == DATA_TYPE_PURE:
        return './data/pure_title/train.csv'
    else:
        return './data/train.tsv'


def get_dev_path(data_type=''):
    if data_type == DATA_TYPE_VALID:
        return './dataset/valid_title/dev.csv'
    if data_type == DATA_TYPE_PURE:
        return './data/pure_title/dev.csv'
    else:
        return './data/dev.tsv'


def get_test_path(data_type=''):
    if data_type == DATA_TYPE_VALID:
        return './dataset/valid_title/test.csv'
    if data_type == DATA_TYPE_PURE:
        return './data/pure_title/test.csv'
    else:
        return './data/test.tsv'


def get_key_title(data_type=''):
    if data_type == DATA_TYPE_VALID:
        return KEY_VALID_TITLE
    else:
        return KEY_PURE_TITLE


def is_short_video_dataset(data_type=''):
    return data_type in [DATA_TYPE_VALID, DATA_TYPE_PURE]


def get_max_len(data_type=''):
    if is_short_video_dataset(data_type):
        return 2048
    else:
        return 512
