import numpy as np
from pydantic import BaseModel, Field
from fastapi import Header
from typing import Optional, List, Union
from vsDa.interfaces.base import app, log
from vsDa.interfaces.definitions.common import Response
from vsDa.core.db import o_faiss, get_index_type
from vsDa.lib.utils import check_tenant


class TrainVectorInput(BaseModel):
    tenant: Optional[str] = Field('_test', description='租户名称')
    index_name: str = Field(description='索引的名称')
    vectors: List[List[float]] = Field(description='向量数据; shape = (数据量，数据的维度) ')
    partition: Optional[str] = Field('', description='索引的分区')


class Result(BaseModel):
    count: int = Field(0, description='插入成功的数量')
    exist_count: int = Field(0, description='重复的数量')
    ids: List[int] = Field([], description='插入的数据的 id 数组')


@app.post('/v1/index/train',
          name="v1 index train",
          response_model=Response,
          description="训练索引")
@log
def index_train(_input: TrainVectorInput, log_id: Union[int, str] = None):
    tenant = _input.tenant
    index_name = _input.index_name
    vectors = _input.vectors
    partition = _input.partition

    tenant = check_tenant(tenant)

    if not index_name:
        return {'code': 0, 'msg': f'index_name 不能为空'}

    if not vectors:
        return {'code': 0, 'msg': f'vectors 不能为空'}

    index = o_faiss.index(tenant, index_name, partition)
    if index is None:
        return {'code': 0, 'msg': f'index "{index_name}({partition})" (tenant: {tenant}) 不存在，请先创建索引'}

    index_type = get_index_type(index)
    if index_type.startswith('Flat'):
        return {'code': 1, 'msg': f'index "{index_name}({partition})" (tenant: {tenant}) 是 {index_type} 索引，不需要训练'}

    vectors = np.array(vectors).astype(np.float32)
    o_faiss.train(tenant, index_name, vectors, partition, log_id=log_id)
    return {'code': 1}


if __name__ == '__main__':
    # 测试代码
    from vsDa.interfaces.index.index_create import index_create

    index_create('test', 384, '')
    index_train(TrainVectorInput(
        index_name='test',
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
    ))
