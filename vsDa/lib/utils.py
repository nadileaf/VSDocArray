import os
import re
import time
import random
import hashlib

_cur_dir = os.path.split(os.path.abspath(__file__))[0]
root_dir = os.path.split(os.path.split(_cur_dir)[0])[0]

reg_not_en_zh = re.compile(r'[^\u3400-\u9FFFa-zA-Z_\-0-9.]')


def uid():
    _id = f'{int(time.time() * 1000)}{int(random.random() * 1000000)}'
    return int(_id)


def md5(_obj):
    text = f'{_obj}'.encode('utf-8')
    m = hashlib.md5()
    m.update(text)
    return m.hexdigest()


def get_relative_dir(*args, root=''):
    """ return the relative path based on the root_dir; if not exists, the dir would be created """
    dir_path = root_dir if not root else root
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    for i, arg in enumerate(args):
        dir_path = os.path.join(dir_path, arg)
        if not os.path.exists(dir_path) and '.' not in arg:
            os.mkdir(dir_path)
    return dir_path


def get_relative_file(*args, root=''):
    return os.path.join(get_relative_dir(*args[:-1], root=root), args[-1])


def check_tenant(tenant: str):
    return md5(tenant) if reg_not_en_zh.search(tenant) else tenant
