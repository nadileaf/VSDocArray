import os
import sys
import time
import inspect
import numpy as np
from functools import wraps

import pandas as pd
from pydantic import BaseModel
from vsDa.lib import utils
from vsDa.config.path import LOG_DIR

LOG = []

MODULE = 'default'
PROCESS = 'default'

LEVEL_MSG = 'MSG'
LEVEL_NOTICE = 'NOTICE'
LEVEL_WARNING = 'WARNING'
LEVEL_ERROR = 'ERROR'

MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_FILE_NO = 10

_incr_id = 100000000


def uid():
    global _incr_id
    _incr_id += 1
    return _incr_id


def fn_name():
    return inspect.stack()[1][3]


def add(_id, function, message, pre_sep='', empty_line=0, _level=LEVEL_MSG, show=True, len_limit=500):
    # construct log message
    _level = f'\033[32;1m{_level}\033[0m' if _level != LEVEL_ERROR else f'\033[31;1m{_level}\033[0m'
    function = f'\033[36;1m{function}\033[0m'

    _time = str(time.strftime('%Y-%m-%d %H:%M:%S'))
    message = message if len(message) < len_limit else message[:len_limit] + '...'
    string = '\n' * empty_line + (
        pre_sep + '\n' if pre_sep else pre_sep) + f'{_id} : {_level} : {_time} : {function} : {message}\n'
    if show:
        print(string[:-1])
        sys.stdout.flush()

    # get correct file path
    dir_path = utils.get_relative_dir(MODULE, PROCESS, root=LOG_DIR)

    file_names = list(filter(lambda x: not x.startswith('.'), os.listdir(dir_path)))
    file_nos = list(map(lambda x: int(x.split('.')[0]), file_names))
    file_nos.sort(reverse=True)

    # 清楚过多的日志文件
    while len(file_nos) > MAX_FILE_NO:
        min_file_no = file_nos.pop()
        os.remove(os.path.join(dir_path, f'{min_file_no}.log'))

    file_no = file_nos[0] if file_nos else 1
    file_path = os.path.join(dir_path, f'{file_no}.log')

    while os.path.exists(file_path) and os.path.getsize(file_path) > MAX_FILE_SIZE:
        file_no += 1
        file_path = os.path.join(dir_path, f'{file_no}.log')

    # write log
    with open(file_path, 'ab') as f:
        f.write(string.encode('utf-8'))


def _process_arg(arg, _iter=3):
    if _iter <= 0:
        return f'{arg}'

    if isinstance(arg, BaseModel):
        arg = arg.__dict__

    if isinstance(arg, pd.DataFrame):
        return len(arg)

    if isinstance(arg, dict):
        new_arg = {}
        num = 0
        for k, v in arg.items():
            if k == 'vectors' and v is not None:
                new_arg[k] = len(v)
            else:
                new_arg[k] = _process_arg(v, _iter - 1)
            num += 1
            if num >= 5:
                break
        arg = new_arg

    elif isinstance(arg, list) or isinstance(arg, tuple):
        return [_process_arg(v, _iter - 1) for v in arg[:5]]

    elif isinstance(arg, np.ndarray):
        arg = list(arg)
        return [_process_arg(v, _iter - 1) for v in arg[:5]]

    return arg


def log(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 生成请求的唯一id
        log_id = uid() if 'log_id' not in kwargs or not kwargs['log_id'] else kwargs['log_id']

        # 记录请求的参数
        log_args = [_process_arg(arg) for arg in args]
        log_kwargs = {k: _process_arg(v) for k, v in kwargs.items()}
        add(log_id, f'Request {func.__name__}', f'args: {log_args}, kwargs: {log_kwargs}')
        s_time = time.time()

        # 执行 函数本身
        kwargs['log_id'] = log_id
        _ret = func(*args, **kwargs)

        e_time = time.time()

        # 修改日志的等级
        _level = LEVEL_MSG
        if isinstance(_ret, dict):
            if 'ret' in _ret and 'log_id' not in _ret:
                _ret['log_id'] = log_id
            if 'ret' in _ret and _ret['ret'] == 0 or 'code' in _ret and _ret['code'] == 0:
                _level = LEVEL_ERROR

        elif isinstance(_ret, int) and _ret == 0:
            _level = LEVEL_ERROR

        use_time = e_time - s_time
        if use_time <= 0.1:
            use_time = f'\033[32;1m{use_time:.4f}s\033[0m'
        elif use_time <= 0.5:
            use_time = f'\033[33;1m{use_time:.4f}s\033[0m'
        elif use_time <= 1:
            use_time = f'\033[34;1m{use_time:.4f}s\033[0m'
        elif use_time <= 5:
            use_time = f'\033[35;1m{use_time:.4f}s\033[0m'
        else:
            use_time = f'\033[31;1m{use_time:.4f}s\033[0m'

        # 记录返回结果
        add(log_id, f'Response {func.__name__}', f'(use time: {use_time}) '
                                                 f'return {_process_arg(_ret)}', _level=_level)
        return _ret

    return wrapper


if __name__ == "__main__":
    print(fn_name())
