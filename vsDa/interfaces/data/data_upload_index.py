import os
import shutil
import zipfile
from typing import Optional, Union
from fastapi import File, UploadFile, Header
from vsDa.interfaces.base import app, log
from vsDa.interfaces.definitions.common import Response
from vsDa.lib.utils import get_relative_file, get_relative_dir
from vsDa.config.path import TMP_DIR, INDEX_DIR


@app.post('/v1/data/upload_index',
          name="v1 data upload index",
          response_model=Response,
          description="上传数据")
@log
def data_upload_index(
        index_file: UploadFile = File(..., description="索引的zip文件"),
        tenant: Optional[str] = Header('_test'),
        log_id: Union[int, str] = None
):
    # 获取文件名
    index_file_name = index_file.filename

    # 读取数据
    index_original_content = index_file.file.read()

    # 保存原文件
    index_file_path = get_relative_file('index', tenant, index_file_name, root=TMP_DIR)
    with open(index_file_path, 'wb') as f:
        f.write(index_original_content)

    if not zipfile.is_zipfile(index_file_path):
        os.remove(index_file_path)
        return {'code': 0, 'msg': f'上传的文件类型必须是 zip'}

    index_tar_dir = get_relative_dir('index', tenant, os.path.splitext(index_file_name)[0], root=TMP_DIR)
    # index_tar_dir = os.path.join(TMP_DIR, 'index', tenant, os.path.splitext(index_file_name)[0])
    index_zip_file = zipfile.ZipFile(index_file_path)
    for file_name in index_zip_file.namelist():
        if not file_name.endswith('.index') and not file_name.endswith('.pkl'):
            continue
        index_zip_file.extract(file_name, index_tar_dir)
    index_zip_file.close()

    for dir_name in os.listdir(index_tar_dir):
        dir_path = os.path.join(index_tar_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        if dir_name == 'index':
            for tmp_tenant in os.listdir(dir_path):
                tmp_tenant_dir = os.path.join(dir_path, tmp_tenant)
                if os.path.isdir(tmp_tenant_dir):
                    for tmp_partition in os.listdir(tmp_tenant_dir):
                        tmp_partition_dir = os.path.join(tmp_tenant_dir, tmp_partition)
                        if os.path.isdir(tmp_partition_dir):
                            for tmp_index_name in os.listdir(tmp_partition_dir):
                                if not tmp_index_name.endswith('.index') and not tmp_index_name.endswith('.pkl'):
                                    continue

                                tmp_index_path = get_relative_file(tmp_tenant, tmp_partition, tmp_index_name,
                                                                   root=INDEX_DIR)
                                if os.path.exists(tmp_index_path):
                                    os.remove(tmp_index_path)
                                shutil.move(os.path.join(tmp_partition_dir, tmp_index_name), tmp_index_path)

        elif dir_name == tenant:
            for tmp_partition in os.listdir(dir_path):
                tmp_partition_dir = os.path.join(dir_path, tmp_partition)
                if os.path.isdir(tmp_partition_dir):
                    for tmp_index_name in os.listdir(tmp_partition_dir):
                        if not tmp_index_name.endswith('.index') and not tmp_index_name.endswith('.pkl'):
                            continue

                        tmp_index_path = get_relative_file(dir_name, tmp_partition, tmp_index_name, root=INDEX_DIR)
                        if os.path.exists(tmp_index_path):
                            os.remove(tmp_index_path)
                        shutil.move(os.path.join(tmp_partition_dir, tmp_index_name), tmp_index_path)

        else:
            for tmp_index_name in os.listdir(dir_path):
                if not tmp_index_name.endswith('.index') and not tmp_index_name.endswith('.pkl'):
                    continue
                tmp_index_path = get_relative_file(tenant, tmp_index_name, root=INDEX_DIR)
                if os.path.exists(tmp_index_path):
                    os.remove(tmp_index_path)
                shutil.move(os.path.join(dir_path, tmp_index_name), tmp_index_path)
            shutil.move(dir_path, get_relative_dir(tenant, root=INDEX_DIR))

    shutil.rmtree(index_tar_dir)
    os.remove(index_file_path)

    return {'code': 1}
