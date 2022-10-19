import os
import sys
import numpy as np
from pydantic import BaseModel, Field
from fastapi import Header
from typing import Optional, List, Any, Union
from vsDa.interfaces.base import app, log
from vsDa.interfaces.definitions.common import Response
from vsDa.core.db import o_faiss
from vsDa.lib.utils import check_tenant


class SearchInput(BaseModel):
    tenants: Optional[List[str]] = Field([], description='租户名称')
    index_names: List[str] = Field(description='索引的名称')
    vectors: List[List[float]] = Field(description='向量数据; shape = (数据量，数据的维度) ')
    partitions: Optional[List[str]] = Field(None, description='索引的分区')
    nprobe: Optional[int] = Field(10, description='查找索引中最近的 nprobe 个中心点')
    top_k: Optional[int] = Field(20, description='返回 top_k 个结果')
    each_top_k: Optional[int] = Field(20, description='多个 index 时，每个 index 返回 top_k 个结果')
    use_mv: Optional[bool] = Field(True, description='搜索时是否考虑滑动平均向量因素; 可用于过滤脏数据')


class Result(BaseModel):
    data: Any = Field(description='插入成功的数量')
    score: float = Field(0, description='该结果的 相似度 或 score')
    mv_score: Optional[float] = Field(0, description="滑动平均向量计算得出的 score")


class SearchResponse(Response):
    data: List[List[Result]] = Field(description='插入数据的结果')


@app.post('/v1/index/search',
          name="v1 index search",
          response_model=SearchResponse,
          description="搜索数据")
@log
def index_search(_input: SearchInput, log_id: Union[int, str] = None):
    tenants = _input.tenants
    index_names = _input.index_names
    vectors = _input.vectors
    partitions = _input.partitions
    nprobe = _input.nprobe
    top_k = _input.top_k
    each_top_k = _input.each_top_k
    use_mv = _input.use_mv

    tenants = tenants if tenants else ['_test'] * len(index_names)
    tenants = list(map(check_tenant, tenants))

    if not index_names:
        return {'code': 0, 'msg': f'index_names 不能为空'}

    if not vectors:
        return {'code': 0, 'msg': f'vectors 不能为空'}

    vectors = np.array(vectors).astype(np.float32)
    _ret = o_faiss.search(vectors, tenants, index_names, partitions, nprobe, top_k, each_top_k,
                          use_mv=use_mv, log_id=log_id)
    return {'code': 1, 'data': _ret}


if __name__ == '__main__':
    # 测试代码
    from vsDa.interfaces.index.index_create import index_create
    from vsDa.interfaces.index.index_train import index_train, TrainVectorInput
    from vsDa.interfaces.index.index_add_vectors import index_add_vectors, VectorInput

    index_create('test', 384, 'aaa', 1000)
    index_create('test', 384, 'bbb', 1000)

    index_train(TrainVectorInput(
        index_name='test',
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
        partition='aaa',
    ))

    index_train(TrainVectorInput(
        index_name='test',
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
        partition='bbb',
    ))

    index_add_vectors(VectorInput(
        index_name='test',
        texts=[f'{i}' for i in range(200)],
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
        info=[{'value': i} for i in range(0, 200)],
        partition='aaa',
    ))
    index_add_vectors(VectorInput(
        index_name='test',
        texts=[f'{i}' for i in range(200)],
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
        info=[{'value': i} for i in range(0, 200)],
        partition='bbb',
    ))

    a = np.zeros([3, 384])
    for i in range(len(a)):
        a[i, :i + 1] = 1
    a = a / np.sqrt(np.sum(a ** 2, axis=1))[..., None]
    #
    # index_add_vectors(VectorInput(
    #     index_name='test',
    #     texts=[f'{i}' for i in range(3)],
    #     vectors=list(map(lambda l: list(map(float, l)), a)),
    #     info=[{'value': i} for i in range(3)],
    #     # filter_exist=True,
    # ))

    # index_add_vectors(VectorInput(
    #     index_name='test',
    #     texts=[f'{i}' for i in range(400, 600)],
    #     vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
    #     info=[{'value': i} for i in range(1200, 1400)],
    # ))

    iii = o_faiss.indices

    ret = index_search(SearchInput(
        index_names=['test', 'test'],
        vectors=list(map(lambda l: list(map(float, l)), a)),
        top_k=4,
        partitions=['aaa', 'bbb'],
        use_mv=False
    ))

    for v_list in ret['data']:
        print('\n--------------------------------------')
        for v in v_list:
            print(v)
