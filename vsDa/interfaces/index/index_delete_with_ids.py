import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, List, Union
from vsDa.interfaces.base import app, log
from vsDa.interfaces.definitions.common import Response
from vsDa.core.db import o_faiss
from vsDa.lib.utils import check_tenant


class IdsInput(BaseModel):
    tenant: Optional[str] = Field('_test', description='租户名称')
    index_name: str = Field(description='索引的名称')
    ids: List[Union[int, str]] = Field(description='数据的 ids, 之前 add vectors 时返回的 ids')
    partition: Optional[str] = Field('', description='索引的分区')


@app.post('/v1/index/delete_with_ids',
          name="v1 index delete with ids",
          response_model=Response,
          description="删除数据，根据 id 删除, id 为 add vector 时返回的 id")
@log
def index_delete_with_ids(_input: IdsInput, log_id: Union[int, str] = None):
    tenant = _input.tenant
    index_name = _input.index_name
    ids = _input.ids
    partition = _input.partition

    tenant = check_tenant(tenant)

    if not index_name:
        return {'code': 0, 'msg': f'index_name 不能为空'}

    if not ids:
        return {'code': 0, 'msg': f'ids 不能为空'}

    o_faiss.delete_with_id(ids, tenant, index_name, partition, log_id=log_id)
    return {'code': 1}


if __name__ == '__main__':
    # 测试代码
    from vsDa.interfaces.index.index_create import index_create
    from vsDa.interfaces.index.index_train import index_train, TrainVectorInput
    from vsDa.interfaces.index.index_add_vectors import index_add_vectors, VectorInput
    from vsDa.interfaces.index.index_search import index_search, SearchInput

    index_create('test11', 384, '', 1000)

    index_train(TrainVectorInput(
        index_name='test11',
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
    ))

    ret_add = index_add_vectors(VectorInput(
        index_name='test11',
        texts=[f'{i}' for i in range(200)],
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
        info=[{'value': i} for i in range(0, 200)],
    ))

    index_delete_with_ids(IdsInput(
        index_name='test11',
        ids=ret_add['data']['ids'][:3],
    ))

    ret = index_search(SearchInput(
        index_names=['test11'],
        vectors=list(map(lambda l: list(map(float, l)), np.eye(5, 384))),
        top_k=4,
    ))

    for v_list in ret['data']:
        print('\n--------------------------------------')
        for v in v_list:
            print(v)
