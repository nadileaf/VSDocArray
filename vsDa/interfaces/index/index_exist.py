import os
from vsDa.config.path import INDEX_DIR
from pydantic import Field
from fastapi import Query
from typing import Optional, Union
from vsDa.interfaces.base import app, log
from vsDa.interfaces.definitions.common import Response
from vsDa.core.db import o_faiss
from vsDa.lib.utils import check_tenant


class ExistResponse(Response):
    data: Optional[bool] = Field(description='是否存在索引')


@app.get('/v1/index/exist',
         name="v1 index exist",
         response_model=ExistResponse,
         description="检查索引是否存在")
@log
def index_exist(
        index_name: str = Query('', description='索引的名称'),
        partition: Optional[str] = Query('', description='索引的分区'),
        tenant: Optional[str] = Query('_test', description='租户名称'),
        log_id: Union[int, str] = None,
):
    index_name = index_name if isinstance(index_name, str) else index_name.default
    partition = partition if isinstance(partition, str) else partition.default
    tenant = tenant if isinstance(tenant, str) else tenant.default

    tenant = check_tenant(tenant)

    partition = partition if partition else o_faiss.DEFAULT

    # check in disk
    index_path = os.path.join(INDEX_DIR, tenant, index_name, f'{partition}.index')
    if os.path.exists(index_path):
        return {'code': 1, 'data': True}

    index = o_faiss.index(tenant, index_name, partition)
    return {'code': 1, 'data': index is not None}


if __name__ == '__main__':
    # 测试代码
    index_exist('test', '', '')
