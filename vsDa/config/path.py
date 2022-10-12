import os

# 代码的根目录
CODE_ROOT = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]

# 代码项目名
PRJ_NAME = os.path.split(CODE_ROOT)[1]

# 数据的根目录
DATA_ROOT = os.path.join('/data', PRJ_NAME) if os.path.exists('/data') else os.path.join(CODE_ROOT, 'data')

# 索引的目录
INDEX_DIR = os.path.join(DATA_ROOT, 'index')

# 索引的目录
SQLITE_DIR = os.path.join(DATA_ROOT, 'sqlite')

# 日志的目录
LOG_DIR = os.path.join(DATA_ROOT, 'logs')

TMP_DIR = os.path.join(DATA_ROOT, 'tmp')

# 创建数据的根目录
_root = r'/'
for dir_name in DATA_ROOT.split(r'/'):
    if not dir_name:
        continue

    _root = os.path.join(_root, dir_name)
    if not os.path.exists(_root):
        os.mkdir(_root)

# 创建目录
for _dir_path in [INDEX_DIR, LOG_DIR, SQLITE_DIR, TMP_DIR]:
    if not os.path.exists(_dir_path):
        os.mkdir(_dir_path)

print(f'DATA_ROOT: {DATA_ROOT}')
