import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Union
from vsDa.interfaces.base import app, log
from vsDa.interfaces.definitions.common import Response
from vsDa.core.db import o_faiss
from vsDa.lib.utils import check_tenant


class VectorInput(BaseModel):
    tenant: Optional[str] = Field('_test', description='租户名称')
    index_name: str = Field(description='索引的名称')
    partition: Optional[str] = Field('', description='索引的分区')

    vectors: List[List[float]] = Field(description='向量数据; shape = (数据量，数据的维度) ')
    ids: Optional[List[int]] = Field([], description='数据的ids')

    texts: Optional[List[Any]] = Field([], description='向量数据对应的文本或描述；与 vector 有一一对应关系，用于生成 id; '
                                                       '若不提供，则以 vector 本身进行去重，但耗时更久')
    info: Optional[List[Any]] = Field([], description='结构化数据 或 文本数据 等; len(info) = 数据量')

    mv_partition: Optional[str] = Field('', description='给 default partition 用 mv 用')
    filter_exist: Optional[bool] = Field(False, description='是否过滤重复')
    ret_id: Optional[bool] = Field(False, description='是否返回数据的id')


class InsertResponse(Response):
    data: Optional[list] = Field(description='插入数据的结果')


@app.post('/v1/index/add_vectors',
          name="v1 index add vectors",
          response_model=InsertResponse,
          description="添加向量到索引 (内存；需要手动调用 save 接口才会将 索引数据保存到磁盘)")
@log
def index_add_vectors(_input: VectorInput, log_id: Union[int, str] = None):
    tenant = _input.tenant
    index_name = _input.index_name
    vectors = _input.vectors
    ids = _input.ids
    texts = _input.texts
    info = _input.info
    partition = _input.partition
    mv_partition = _input.mv_partition
    filter_exist = _input.filter_exist
    ret_id = _input.ret_id

    tenant = check_tenant(tenant)

    if not index_name:
        return {'code': 0, 'msg': f'index_name 不能为空'}

    if not vectors:
        return {'code': 0, 'msg': f'vectors 不能为空'}

    if not ids and not texts:
        return {'code': 0, 'msg': f'ids 和 texts 不能都为空'}

    index = o_faiss.index(tenant, index_name, partition)
    if index is None:
        return {'code': 0, 'msg': f'index "{index_name}({partition})" (tenant: "{tenant}") 不存在，请先创建索引'}

    if not index.is_trained:
        return {'code': 0, 'msg': f'index "{index_name}({partition})" (tenant: {tenant}) 还没 train，需要先调用 train 接口'}

    vectors = np.array(vectors).astype(np.float32)

    if not ids:
        in_ids, filter_indices, doc_ids = o_faiss.add_data(tenant, index_name, partition, texts, info, filter_exist)
        if in_ids is None:
            return {'code': 1, 'msg': '数据已存在'}

        if len(filter_indices) < len(vectors):
            vectors = vectors[filter_indices]
    else:
        doc_ids = ids
        in_ids: np.ndarray = np.array(ids, dtype=np.int32)

    o_faiss.add(tenant, index_name, partition, in_ids, vectors, mv_partition=mv_partition, log_id=log_id)

    return {'code': 1, 'data': doc_ids}


if __name__ == '__main__':
    # 测试代码
    from vsDa.interfaces.index.index_create import index_create
    from vsDa.interfaces.index.index_train import index_train, TrainVectorInput

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
        filter_exist=False,
    ))
