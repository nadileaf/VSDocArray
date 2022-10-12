import numpy as np
from pydantic import BaseModel, Field
from fastapi import Header
from typing import Optional, List, Any, Union
from vsDa.interfaces.base import app, log
from vsDa.interfaces.definitions.common import Response
from vsDa.core.db import o_faiss
from vsDa.lib.utils import check_tenant


class UpdateInfoInput(BaseModel):
    tenant: Optional[str] = Field('_test', description='租户名称')
    index_name: str = Field(description='索引的名称')
    vectors: List[List[float]] = Field(description='向量数据; shape = (数据量，数据的维度) ')
    texts: Optional[List[Any]] = Field(description='向量数据对应的文本或描述；与 vector 有一一对应关系，用于生成 id; '
                                                   '若不提供，则以 vector 本身进行去重，但耗时更久')
    old_info: List[Any] = Field([], description='结构化数据 或 文本数据 等; len(info) = 数据量')
    new_info: List[Any] = Field([], description='结构化数据 或 文本数据 等; len(info) = 数据量')
    partition: Optional[str] = Field('', description='索引的分区')


@app.post('/v1/index/update_with_info',
          name="v1 index update with info",
          response_model=Response,
          description="删除数据，根据 info 删除")
@log
def index_update_with_info(_input: UpdateInfoInput, log_id: Union[int, str] = None):
    tenant = _input.tenant
    index_name = _input.index_name
    vectors = _input.vectors
    texts = _input.texts
    old_info = _input.old_info
    new_info = _input.new_info
    partition = _input.partition

    tenant = check_tenant(tenant)

    if not index_name:
        return {'code': 0, 'msg': f'index_name 不能为空'}

    if not texts and not vectors:
        return {'code': 0, 'msg': f'texts, vectors 不能为空'}

    vectors = np.array(vectors).astype(np.float32)
    o_faiss.update_with_info(tenant, index_name, vectors, texts, old_info, new_info, partition, log_id=log_id)
    return {'code': 1}


if __name__ == '__main__':
    # 测试代码
    from vsDa.interfaces.index.index_create import index_create
    from vsDa.interfaces.index.index_train import index_train, TrainVectorInput
    from vsDa.interfaces.index.index_add_vectors import index_add_vectors, VectorInput
    from vsDa.interfaces.index.index_search import index_search, SearchInput

    index_create('test11', 384, '', 2000)

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

    index_update_with_info(UpdateInfoInput(
        index_name='test11',
        texts=[f'{i}' for i in range(3)],
        vectors=list(map(lambda l: list(map(float, l)), np.eye(3, 384))),
        old_info=[{'value': i} for i in range(0, 3)],
        new_info=[{'value': i} for i in range(3, 6)],
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
