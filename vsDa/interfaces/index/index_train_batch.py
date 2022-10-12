import numpy as np
from pydantic import BaseModel, Field
from fastapi import Header
from typing import Optional, List, Union
from vsDa.interfaces.base import app, log
from vsDa.interfaces.definitions.common import Response
from vsDa.core.db import o_faiss
from vsDa.lib.utils import check_tenant

# 用于缓存数据
d_key_2_vectors = {}


class VectorInput(BaseModel):
    tenant: Optional[str] = Field('_test', description='租户名称')
    index_name: str = Field(description='索引的名称')
    vectors: Optional[List[List[float]]] = Field(description='向量数据; shape = (数据量，数据的维度) ')
    partition: Optional[str] = Field('', description='索引的分区')
    start_train: Optional[bool] = Field(False, description='是否开始训练；false 则先缓存数据，等数据齐了再train；true 则开始训练')


class Result(BaseModel):
    count: int = Field(0, description='插入成功的数量')
    exist_count: int = Field(0, description='重复的数量')
    ids: List[int] = Field([], description='插入的数据的 id 数组')


@app.post('/v1/index/train_batch',
          name="v1 index train batch",
          response_model=Response,
          description="训练索引；避免数据过大，没法经过网络请求传输")
@log
def index_train_batch(_input: VectorInput, log_id: Union[int, str] = None):
    tenant = _input.tenant
    index_name = _input.index_name
    vectors = _input.vectors
    partition = _input.partition
    start_train = _input.start_train

    tenant = check_tenant(tenant)

    if not index_name:
        return {'code': 0, 'msg': f'index_name 不能为空'}

    if not start_train and not vectors:
        return {'code': 0, 'msg': f'vectors 不能为空'}

    key = f'{tenant}____{index_name}____{partition}'
    if key not in d_key_2_vectors:
        d_key_2_vectors[key] = []

    if not start_train:
        total_len = 0
        for key, v in d_key_2_vectors.items():
            total_len += np.sum(list(map(lambda x: len(x), v)))

        # 控制内存泄露问题
        if total_len + len(vectors) >= 1000000:
            return {'code': 0, 'msg': f'当前缓存数据量过大，请稍后再使用'}

        d_key_2_vectors[key].append(vectors)

    else:
        index = o_faiss.index(tenant, index_name, partition)
        if index is None:
            return {'code': 0, 'msg': f'index "{index_name}({partition})" (tenant: {tenant}) 不存在，请先创建索引'}
        if not d_key_2_vectors[key]:
            return {'code': 0, 'msg': f'index "{index_name}({partition})" (tenant: {tenant}) 没有数据'}

        vectors = np.vstack(d_key_2_vectors[key]).astype(np.float32)
        del d_key_2_vectors[key]
        o_faiss.train(tenant, index_name, vectors, partition, log_id=log_id)

    return {'code': 1}


if __name__ == '__main__':
    # 测试代码
    from vsDa.interfaces.index.index_create import index_create

    index_create('test', 384, '')
    index_train_batch(VectorInput(
        index_name='test',
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
        start_train=False,
    ))

    index_train_batch(VectorInput(
        index_name='test',
        vectors=list(map(lambda l: list(map(float, l)), np.random.rand(50, 384))),
        start_train=False,
    ))

    index_train_batch(VectorInput(
        index_name='test',
        vectors=None,
        start_train=True,
    ))
