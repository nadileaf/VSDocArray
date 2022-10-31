from fastapi import Query
from typing import Optional, Union
from vsDa.interfaces.base import app, log
from vsDa.interfaces.definitions.common import Response
from vsDa.core.db import o_faiss
from vsDa.lib.utils import check_tenant


@app.get('/v1/index/load',
         name="v1 index load",
         response_model=Response,
         description="加载索引")
@log
def index_load(
        index_name: str = Query('', description='索引的名称; 若为 * , 则加载索引索引'),
        partition: Optional[str] = Query('', description='索引的分区'),
        tenant: Optional[str] = Query('_test', description='租户名称'),
        refresh: Optional[str] = Query(False, description='是否刷新已在内存中的索引'),
        log_id: Union[int, str] = None,
):
    tenant = tenant if isinstance(tenant, str) else tenant.default
    index_name = index_name if isinstance(index_name, str) else index_name.default
    partition = partition if isinstance(partition, str) else partition.default
    refresh = refresh if isinstance(refresh, bool) else refresh.default

    tenant = check_tenant(tenant)

    if not index_name:
        return {'code': 0, 'msg': f'index_name 不能为空'}

    _ret = 1
    if index_name == '*':
        o_faiss.load(tenant, log_id=log_id)

    else:
        _ret = o_faiss.load_one(tenant, index_name, partition, refresh=refresh, log_id=log_id)

    msg = 'Successfully' if _ret else 'Fail'
    msg = f'{msg} loading index "{index_name}({partition})" (tenant: "{tenant}")'
    return {'code': _ret, 'msg': msg}


if __name__ == '__main__':
    # 测试代码
    import numpy as np
    from vsDa.interfaces.index.index_search import index_search, SearchInput

    index_load('test', '')

    ret = index_search(SearchInput(
        index_names=['test'],
        vectors=list(map(lambda l: list(map(float, l)), np.eye(3, 384))),
        top_k=3,
    ))

    for v_list in ret['data']:
        print('\n--------------------------------------')
        for v in v_list:
            print(v)
