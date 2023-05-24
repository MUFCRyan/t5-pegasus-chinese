KEY_VALID_TITLE = 'valid_title'
KEY_PURE_TITLE = 'pure_title'
KEY_PHOTO_ID = 'photo_id'
KEY_SUMMARY = 'summary'


DATA_TYPE_VALID = KEY_VALID_TITLE
DATA_TYPE_PURE = KEY_PURE_TITLE


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
        return KEY_PURE_TITLE


def is_short_video_dataset(data_type=''):
    return data_type in [DATA_TYPE_VALID, DATA_TYPE_PURE]


def get_bz_max_len(data_type=''):
    if is_short_video_dataset(data_type):
        return 2, 1536
    else:
        return 16, 512
