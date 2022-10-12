from fastapi import Query
from typing import Optional, Union
from vsDa.interfaces.base import app, log
from vsDa.interfaces.definitions.common import Response
from vsDa.core.db import o_faiss
from vsDa.lib.utils import check_tenant


@app.get('/v1/index/save',
         name="v1 index save",
         response_model=Response,
         description="保存索引到文件")
@log
def index_save(
        index_name: str = Query('', description='索引的名称; 若为 * , 则保存所有索引'),
        partition: Optional[str] = Query('', description='索引的分区'),
        tenant: Optional[str] = Query('_test', description='租户名称'),
        log_id: Union[int, str] = None
):
    tenant = tenant if isinstance(tenant, str) else tenant.default
    index_name = index_name if isinstance(index_name, str) else index_name.default
    partition = partition if isinstance(partition, str) else partition.default

    tenant = check_tenant(tenant)

    if not index_name:
        return {'code': 0, 'msg': f'index_name 不能为空'}

    _ret = 1
    if index_name == '*':
        o_faiss.save(tenant, log_id=log_id)
    else:
        _ret = o_faiss.save_one(tenant, index_name, partition, log_id=log_id)

    msg = 'Successfully' if _ret else 'Fail'
    msg = f'{msg} save index "{index_name}({partition})" (tenant: "{tenant}")'

    return {'code': _ret, 'msg': msg}


if __name__ == '__main__':
    # 测试代码
    import numpy as np
    from vsDa.interfaces.index.index_create import index_create
    from vsDa.interfaces.index.index_train import index_train, TrainVectorInput
    from vsDa.interfaces.index.index_add_vectors import index_add_vectors, VectorInput
    from vsDa.interfaces.index.index_release import index_release

    index_release('test')
    index_create('test', 384, '', 500)

    index_train(TrainVectorInput(
        index_name='test',
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
    ))

    index_add_vectors(VectorInput(
        index_name='test',
        texts=[f'{i}' for i in range(200)],
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
        info=[{'value': i} for i in range(200)],
    ))

    index_save('test')
