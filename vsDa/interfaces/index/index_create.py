import re
from fastapi import Query
from typing import Optional, Union
from vsDa.interfaces.base import app, log
from vsDa.interfaces.definitions.common import Response
from vsDa.core.db import o_faiss, get_index
from vsDa.lib.utils import check_tenant

_reg_valid = re.compile(r'^[a-zA-Z0-9_\-. \u3400-\u9FFF]+$')


@app.get('/v1/index/create',
         name="v1 index create",
         response_model=Response,
         description="创建索引 (暂时只支持 metric: 内积，若需更多 metric，请联系林雨森)")
@log
def index_create(
        index_name: str = Query('', description='索引的名称'),
        dim_size: int = Query(0, description='向量的维度大小, 如 384, 1024 等'),
        partition: Optional[str] = Query('', description='索引的分区'),
        count: Optional[int] = Query(10000, description='预估的数据量'),
        tenant: Optional[str] = Query('_test', description='租户名称'),
        n_list: Optional[int] = Query(None, description='cluster size'),
        log_id: Union[int, str] = None,
):
    index_name = index_name if isinstance(index_name, str) else index_name.default
    dim_size = int(dim_size) if isinstance(dim_size, int) else dim_size.default
    partition = partition if isinstance(partition, str) else partition.default
    count = int(count) if isinstance(count, int) else count.default
    tenant = tenant if isinstance(tenant, str) else tenant.default
    n_list = n_list if isinstance(n_list, int) or n_list is None else n_list.default

    tenant = check_tenant(tenant)

    if not index_name:
        return {'code': 0, 'msg': f'index_name 不能为空'}
    if not dim_size:
        return {'code': 0, 'msg': f'dim_size 不能为空 或 0'}
    if not _reg_valid.search(index_name) or (partition and not _reg_valid.search(partition)) or \
            not _reg_valid.search(tenant):
        return {'code': 0, 'msg': f'index_name, partition, tenant 都需要符合 pattern: "{_reg_valid.pattern}"'}

    partition = partition if partition else o_faiss.DEFAULT

    if tenant not in o_faiss.indices:
        o_faiss.indices[tenant] = {}
    if index_name not in o_faiss.indices[tenant]:
        o_faiss.indices[tenant][index_name] = {}
    if partition in o_faiss.indices[tenant][index_name]:
        return {'code': 1, 'msg': f'index_name "{index_name}({partition})" (tenant: {tenant}) 已存在，请先删除索引'}

    o_faiss.indices[tenant][index_name][partition] = get_index(count, dim_size, n_list)

    return {'code': 1}


if __name__ == '__main__':
    # 测试代码
    index_create('test', 384, '', 500)
