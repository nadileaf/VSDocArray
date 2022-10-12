import os
import shutil
import zipfile
from typing import Optional, Union
from fastapi import File, UploadFile, Header
from vsDa.interfaces.base import app, log
from vsDa.interfaces.definitions.common import Response
from vsDa.lib.utils import get_relative_file, get_relative_dir
from vsDa.config.path import TMP_DIR, SQLITE_DIR
from vsDa.core.db import o_faiss


@app.post('/v1/data/upload_sqlite',
          name="v1 data upload sqlite",
          response_model=Response,
          description="上传数据")
@log
def data_upload_sqlite(
        sqlite_file: UploadFile = File(..., description="数据的zip文件"),
        tenant: Optional[str] = Header('_test'),
        log_id: Union[int, str] = None
):
    # 获取文件名
    sqlite_file_name = sqlite_file.filename

    # 读取数据
    sqlite_original_content = sqlite_file.file.read()

    # 保存原文件
    sqlite_file_path = get_relative_file('sqlite', tenant, sqlite_file_name, root=TMP_DIR)
    with open(sqlite_file_path, 'wb') as f:
        f.write(sqlite_original_content)

    if not zipfile.is_zipfile(sqlite_file_path):
        os.remove(sqlite_file_path)
        return {'code': 0, 'msg': f'上传的文件类型必须是 zip'}

    sqlite_tar_dir = get_relative_dir('sqlite', tenant, os.path.splitext(sqlite_file_name)[0], root=TMP_DIR)
    sqlite_zip_file = zipfile.ZipFile(sqlite_file_path)
    for file_name in sqlite_zip_file.namelist():
        if '.sqlite' not in file_name:
            continue
        sqlite_zip_file.extract(file_name, sqlite_tar_dir)
    sqlite_zip_file.close()

    for file_name in os.listdir(sqlite_tar_dir):
        file_path = os.path.join(sqlite_tar_dir, file_name)
        if os.path.isdir(file_path):
            for sub_file_name in os.listdir(file_path):
                if '.sqlite' not in sub_file_name:
                    continue
                shutil.move(os.path.join(file_path, sub_file_name), os.path.join(SQLITE_DIR, sub_file_name))
        elif '.sqlite' in file_name:
            shutil.move(file_path, os.path.join(SQLITE_DIR, file_name))

    shutil.rmtree(sqlite_tar_dir)
    os.remove(sqlite_file_path)

    return {'code': 1}
