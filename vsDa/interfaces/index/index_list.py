import os
from pydantic import Field
from fastapi import Query
from typing import Optional, Union
from vsDa.interfaces.base import app, log
from vsDa.interfaces.definitions.common import Response
from vsDa.core.db import o_faiss, get_metric, get_index_type
from vsDa.config.path import INDEX_DIR
from vsDa.lib.utils import check_tenant


class IndicesResponse(Response):
    data: Optional[dict] = Field({}, description='索引列表 dict')


@app.get('/v1/index/list',
         name="v1 index list",
         response_model=IndicesResponse,
         description="列举索引")
@log
def index_list(tenant: Optional[str] = Query('_test', description='租户名称'), log_id: Union[int, str] = None):
    tenant = tenant if isinstance(tenant, str) else tenant.default

    tenant = check_tenant(tenant)

    if tenant not in o_faiss.indices:
        return {'code': 1, 'data': {}, 'msg': f'tenant "{tenant}" 下没有索引'}

    indices = {}
    for index_name, partitions in o_faiss.indices[tenant].items():
        indices[index_name] = {}
        for partition, _index in partitions.items():
            indices[index_name][partition] = {
                'index_name': index_name,
                'partition': partition,
                'storage_type': 'in_memory',
                'dim_size': _index.d,
                'n_total': _index.ntotal,
                'is_trained': _index.is_trained,
                'metric_type': get_metric(_index.metric_type),
                'index_type': get_index_type(_index),
            }

    index_dir = os.path.join(INDEX_DIR, tenant)
    if os.path.isdir(index_dir):
        for index_name in os.listdir(index_dir):
            partition_dir = os.path.join(INDEX_DIR, index_name)
            if not os.path.isdir(partition_dir):
                continue

            if index_name not in indices:
                indices[index_name] = {}

            for file_name in os.listdir(partition_dir):
                if not file_name.endswith('.index'):
                    continue

                partition = file_name[:-len('.index')]
                if partition not in indices[index_name]:
                    indices[index_name][partition] = {
                        'index_name': index_name,
                        'partition': partition,
                        'storage_type': 'in_disk'
                    }

    return {'code': 1, 'data': indices}


if __name__ == '__main__':
    # 测试代码
    import json

    ret = index_list()
    print(json.dumps(ret, ensure_ascii=False, indent=2))
