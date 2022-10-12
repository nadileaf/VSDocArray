from pydantic import Field
from fastapi import Header, Query
from typing import Optional, Union
from vsDa.interfaces.base import app, log
from vsDa.interfaces.definitions.common import Response
from vsDa.core.db import o_faiss
from vsDa.lib.utils import check_tenant


class InfoResponse(Response):
    data: Optional[list] = Field([], description='索引存储的数据的列表')


@app.get('/v1/index/info',
         name="v1 index info",
         response_model=InfoResponse,
         description="索引存储的数据 列表")
@log
def index_info(
        index_name: str = Query('', description='索引的名称; 若为 * , 则加载索引索引'),
        partition: Optional[str] = Query('', description='索引的分区'),
        tenant: Optional[str] = Header('_test'),
        log_id: Union[int, str] = None,
):
    tenant = tenant if isinstance(tenant, str) else tenant.default
    index_name = index_name if isinstance(index_name, str) else index_name.default
    partition = partition if isinstance(partition, str) else partition.default

    tenant = check_tenant(tenant)

    if tenant not in o_faiss.indices:
        return {'code': 1, 'data': {}, 'msg': f'tenant "{tenant}" 下没有索引'}

    data = o_faiss.list_info(tenant, index_name, partition)

    return {'code': 1, 'data': data}


if __name__ == '__main__':
    # 测试代码
    import json
    import numpy as np
    from vsDa.interfaces.index.index_create import index_create
    from vsDa.interfaces.index.index_train import index_train, TrainVectorInput
    from vsDa.interfaces.index.index_add_vectors import index_add_vectors, VectorInput

    index_create('test', 384, '', 200)

    index_train(TrainVectorInput(
        index_name='test',
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
    ))

    index_add_vectors(VectorInput(
        index_name='test',
        texts=[f'{i}' for i in range(600, 800)],
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
        info=[{'value': i} for i in range(200, 400)],
    ))

    ret = index_info('test')
    print(json.dumps(ret, ensure_ascii=False, indent=2))
