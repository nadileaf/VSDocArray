import os
import math
import faiss
import threading
import numpy as np
from docarray import Document, DocumentArray
from typing import List, Union
from six.moves import cPickle as pickle
from sklearn.metrics.pairwise import cosine_similarity
from vsDa.config.path import INDEX_DIR, SQLITE_DIR
from vsDa.lib.utils import md5, get_relative_file
from vsDa.lib import logs


class Faiss:
    DEFAULT = '__default'

    def __init__(self, data_dir: str = None, index_dir: str = None):
        self.__data_dir = data_dir if data_dir and os.path.isdir(data_dir) else SQLITE_DIR
        self.__idx_dir = index_dir if index_dir and os.path.isdir(index_dir) else INDEX_DIR

        # 记录索引
        self.indices = {}

        # 记录 index 每个分区的滑动平均向量
        self.mv_indices = {}

    def get_da(self, tenant: str, index_name: str, partition: str = ''):
        table_name = get_table_name(tenant, index_name, partition)
        file_name = f'{table_name}.sqlite'
        file_path = os.path.join(self.__data_dir, file_name)
        return DocumentArray(storage='sqlite', config={'connection': file_path, 'table_name': 'tb'})

    def index(self, tenant: str, index_name: str, partition: str = '') -> Union[None, faiss.Index]:
        if tenant not in self.indices or index_name not in self.indices[tenant] or \
                (partition and partition not in self.indices[tenant][index_name]):
            return
        partition = partition if partition else self.DEFAULT
        return self.indices[tenant][index_name][partition]

    @logs.log
    def train(self, tenant: str, index_name: str, vectors: np.ndarray, partition: str = '', log_id=None):
        index = self.index(tenant, index_name, partition)
        if index is not None:
            index.train(vectors)

    @logs.log
    def add_data(self,
                 tenant: str,
                 index_name: str,
                 partition: str,
                 texts: List[str],
                 info: list,
                 filter_exist: bool = True,
                 log_id=None):
        docs = [Document(
            id=md5((text, f'{info}', partition)),
            text=text,
            tags=info[i],
        ) for i, text in enumerate(texts)]

        da: DocumentArray = self.get_da(tenant, index_name, partition)

        # 过滤重复数据
        d_id = {}
        for i, d in enumerate(docs):
            if d.id not in da and d.id not in d_id:
                d_id[d.id] = i
        filter_indices = list(d_id.values())

        filter_docs = list(map(lambda i: docs[i], filter_indices))

        # 若都是重复数据
        if not filter_docs:
            # 若需要重复插入
            if not filter_exist:
                mid_list = list(map(lambda x: x.id, docs))
                ids = da[mid_list, 'tags__idx']
                return np.array(ids, dtype=np.int32), list(range(len(docs))), mid_list

            return None, [], []

        # 添加位置标记
        length = len(da)
        for i, doc in enumerate(filter_docs):
            doc.tags['idx'] = length + i

        # 用于 encode for vector
        filter_texts = list(map(lambda x: x.text, filter_docs))

        doc_ids = list(map(lambda x: x.id, filter_docs))

        # 插入数据
        with da:
            da.extend(filter_docs)

        # 返回 ids
        length = len(da)
        return np.arange(length, length - len(filter_texts), -1), filter_indices, doc_ids

    @logs.log
    def add(self,
            tenant: str,
            index_name: str,
            partition: str,
            ids: np.ndarray,
            vectors: np.ndarray,
            mv_partition='',
            log_id=None) -> dict:
        """ 插入数据到 index，返回 插入成功的数量 insert_count """
        partition = partition if partition else self.DEFAULT

        index = self.index(tenant, index_name, partition)
        index_type = get_index_type(index)

        # 添加 到 index
        if index_type.startswith('Flat'):
            index.add(vectors)
        else:
            index.add_with_ids(vectors, ids)

        # 若有 partition，记录该 partition 的滑动平均向量
        if mv_partition or partition and partition != self.DEFAULT:
            tmp_partition = mv_partition if mv_partition else partition
            if tenant not in self.mv_indices:
                self.mv_indices[tenant] = {}
            if index_name not in self.mv_indices[tenant]:
                self.mv_indices[tenant][index_name] = {}
            if tmp_partition not in self.mv_indices[tenant][index_name]:
                self.mv_indices[tenant][index_name][tmp_partition] = None
            mv_index = self.mv_indices[tenant][index_name][tmp_partition]

            if not mv_index:
                self.mv_indices[tenant][index_name][tmp_partition] = {
                    'vector': np.mean(vectors, axis=0),
                    'count': len(vectors)
                }

            else:
                count = mv_index['count']
                avg_embedding = mv_index['vector']

                for v in vectors:
                    beta = min(0.001, 2 / (1 + count))
                    count += 1
                    avg_embedding = avg_embedding * (1 - beta) + v * beta

                self.mv_indices[tenant][index_name][tmp_partition] = {'vector': avg_embedding, 'count': count}

    def list_info(self, tenant: str, index_name: str, partition: str = '', log_id=None) -> list:
        da = self.get_da(tenant, index_name, partition)
        if len(da) == 0:
            return []

        ids = da[:, 'id']
        texts = da[:, 'text']
        tags = da[:, 'tags']
        return list({_id: {'text': texts[_i], **tags[_i]} for _i, _id in enumerate(ids)}.items())

    def save_one(self, tenant: str, index_name: str, partition: str = '', log_id=None) -> int:
        _index = self.index(tenant, index_name, partition)
        if _index is None:
            return 0

        if not partition:
            for partition, _index in self.indices[tenant][index_name].items():
                if _index is None:
                    continue

                index_path = get_relative_file(tenant, index_name, f'{partition}.index', root=self.__idx_dir)
                faiss.write_index(_index, index_path)

            if tenant in self.mv_indices and index_name in self.mv_indices[tenant] and \
                    self.mv_indices[tenant][index_name] is not None:
                with open(get_relative_file(tenant, index_name, 'mv_index.pkl', root=self.__idx_dir), 'wb') as f:
                    pickle.dump(self.mv_indices[tenant][index_name], f)

        else:
            index_path = get_relative_file(tenant, index_name, f'{partition}.index', root=self.__idx_dir)
            faiss.write_index(_index, index_path)

            if partition == self.DEFAULT and tenant in self.mv_indices and index_name in self.mv_indices[tenant] and \
                    self.mv_indices[tenant][index_name] is not None:
                with open(get_relative_file(tenant, index_name, 'mv_index.pkl', root=self.__idx_dir), 'wb') as f:
                    pickle.dump(self.mv_indices[tenant][index_name], f)

        return 1

    def save(self, tenant: str, log_id=None):
        """ 保存当前的所有索引到文件里 """
        if tenant not in self.indices:
            logs.add(log_id, 'save', f'tenant "{tenant}" 没有索引')
            return

        for index_name, index in self.indices[tenant].items():
            self.save_one(tenant, index_name, log_id=log_id)

    def load_one(self, tenant: str, index_name: str, partition: str = '', log_id=None) -> int:
        if not partition:
            index_dir = os.path.join(self.__idx_dir, tenant, index_name)
            if (not os.path.isdir(index_dir) or not os.listdir(index_dir)) and \
                    (tenant not in self.indices or index_name not in self.indices[tenant]):
                return 0

            if tenant not in self.indices:
                self.indices[tenant] = {}
            if tenant not in self.mv_indices:
                self.mv_indices[tenant] = {}
            if index_name not in self.indices[tenant]:
                self.indices[tenant][index_name] = {}

            for file_name in os.listdir(index_dir):
                if not file_name.endswith('.index'):
                    continue

                # 若本身已在内存中，无需重复加载
                partition = file_name[:-len('.index')]
                if self.index(tenant, index_name, partition) is not None:
                    continue

                index_path = os.path.join(index_dir, file_name)
                self.indices[tenant][index_name][partition] = faiss.read_index(index_path)

            # 若文件存在 且 没有被加载到内存
            mv_index_path = os.path.join(index_dir, 'mv_index.pkl')
            if os.path.exists(mv_index_path) and (index_name not in self.mv_indices[tenant] or
                                                  self.mv_indices[tenant][index_name] is None):
                with open(mv_index_path, 'rb') as f:
                    self.mv_indices[tenant][index_name] = pickle.load(f)

        else:
            index_path = os.path.join(self.__idx_dir, tenant, index_name, f'{partition}.index')
            if not os.path.exists(index_path) and (
                    tenant not in self.indices or index_name not in self.indices[tenant] or
                    partition not in self.indices[tenant][index_name]):
                return 0

            if tenant not in self.indices:
                self.indices[tenant] = {}
            if tenant not in self.mv_indices:
                self.mv_indices[tenant] = {}
            if index_name not in self.indices[tenant]:
                self.indices[tenant][index_name] = {}

            if self.index(tenant, index_name, partition) is None:
                self.indices[tenant][index_name][partition] = faiss.read_index(index_path)

        return 1

    def load(self, tenant: str, log_id=None):
        """ 从文件中加载索引 """
        _tenant_dir = os.path.join(self.__idx_dir, tenant)
        if not os.path.isdir(_tenant_dir):
            return

        for index_name in os.listdir(_tenant_dir):
            self.load_one(tenant, index_name, log_id=log_id)

    def release(self, tenant: str, index_name: str, partition: str = '', save=True, log_id=None) -> int:
        # release index 前，先保存索引
        if save:
            ret = self.save_one(tenant, index_name, partition, log_id=log_id)
            if not ret:
                return 0

        if not partition:
            if tenant in self.indices and index_name in self.indices[tenant]:
                del self.indices[tenant][index_name]

        else:
            if tenant in self.indices and index_name in self.indices[tenant] and \
                    partition in self.indices[tenant][index_name]:
                del self.indices[tenant][index_name][partition]

        return 1

    def exist(self, tenant: str, index_name: str, partition: str):
        if self.index(tenant, index_name, partition) is not None:
            return True

        partition = partition if partition else self.DEFAULT
        if os.path.exists(os.path.join(self.__idx_dir, tenant, index_name, f'{partition}.index')):
            return True
        return False

    def is_train(self, tenant: str, index_name: str, partition: str):
        index = self.index(tenant, index_name, partition)
        return index.is_trained if index is not None else False

    def create(self, tenant: str, index_name: str, partition: str, count: int, dim_size: int, n_list: int = None):
        partition = partition if partition else self.DEFAULT
        if tenant not in self.indices:
            self.indices[tenant] = {}
        if index_name not in self.indices[tenant]:
            self.indices[tenant][index_name] = {}
        if partition not in self.indices[tenant][index_name]:
            self.indices[tenant][index_name][partition] = get_index(count, dim_size, n_list)

    @logs.log
    def search(self,
               vectors: np.ndarray,
               tenants: List[str],
               index_names: List[str],
               partitions: List[str] = None,
               nprobe=10,
               top_k=20,
               each_top_k=20,
               use_mv=True,
               log_id=None) -> List[List[dict]]:
        if vectors is None or not vectors.any():
            return []

        results = [[] for _ in range(len(vectors))]
        avg_results = [{} for _ in range(len(vectors))]
        d_table_name_2_ids = {}

        partitions = partitions if partitions else [''] * len(index_names)
        for i, index_name in enumerate(index_names):
            tenant = tenants[i]
            partition = partitions[i] if partitions[i] else self.DEFAULT
            self._search_a_index(tenant, index_name, partition, vectors, nprobe, each_top_k,
                                 avg_results, use_mv, d_table_name_2_ids, results, log_id=log_id)

        # 获取具体的结构化信息
        d_table_id_2_info = self._get_info(d_table_name_2_ids, log_id=log_id)

        return _combine_results(results, avg_results, d_table_id_2_info, top_k, log_id=log_id)

    def delete_with_id(self, ids: List[int], tenant: str, index_name: str, partition: str = '', log_id=None):
        da = self.get_da(tenant, index_name, partition)
        if len(da) == 0:
            return

        if not ids:
            return

        if isinstance(ids[0], int):
            ids.sort(reverse=True)
            len_da = len(da)
            ids = list(filter(lambda x: x < len_da, ids))

            for _id in ids:
                del da[_id]

        else:
            ids = list(filter(lambda x: x in da, ids))
            if ids:
                del da[ids]

    @logs.log
    def _search_a_index(self,
                        tenant: str,
                        index_name: str,
                        partition: str,
                        vectors: np.ndarray,
                        nprobe: int,
                        top_k: int,
                        avg_results: List[dict],
                        use_mv: bool,
                        d_table_name_2_ids: dict,
                        results: List[list],
                        log_id=None):
        # 获取 index
        index = self.index(tenant, index_name, partition)
        if index is None:
            return

        table_name = get_table_name(tenant, index_name, partition)

        if use_mv and partition == self.DEFAULT and tenant in self.mv_indices and \
                index_name in self.mv_indices[tenant]:
            # 获取该 index 每个 partition 的 滑动平均向量
            mv_indices = self.mv_indices[tenant][index_name]
            mv_indices = dict(filter(lambda x: x[1], mv_indices.items()))

            tmp_partitions = list(mv_indices.keys())
            avg_vectors = list(map(lambda x: x['vector'], mv_indices.values()))

            # 根据 滑动平均向量，计算语义相似度
            sims = cosine_similarity(vectors, avg_vectors)

            # 整理、排序 滑动平均向量计算得出的结果
            for _j, sim in enumerate(sims):
                sim = list(zip(tmp_partitions, sim))
                sim.sort(key=lambda x: -x[1])
                avg_results[_j][table_name] = dict(sim)

        index.nprobe = nprobe

        D, I = index.search(vectors, top_k)

        if table_name not in d_table_name_2_ids:
            d_table_name_2_ids[table_name] = []
        d_table_name_2_ids[table_name] += list(set(list(map(int, I.reshape(-1)))))

        for _i, _result_ids in enumerate(I):
            similarities = D[_i]
            results[_i] += [
                {'id': _id, 'score': _similarity, 'table_name': table_name}
                for _id, _similarity in set(list(zip(_result_ids, similarities))) if _id != -1
            ]

    def _get_info_thread(self, ids: list, table_name: str, d_table_id_2_info: dict):
        if not ids:
            return

        clean_table_name = table_name[:-9] if table_name.endswith('__default') else table_name
        tenant, index_name, partition = clean_table_name.split('__')
        da = self.get_da(tenant, index_name, partition)

        texts = da[ids, 'text']
        tags = da[ids, 'tags']

        for _i, _id in enumerate(ids):
            info = {'text': texts[_i], **tags[_i]}
            table_id = f"{table_name}__{_id}"
            d_table_id_2_info[table_id] = info

    @logs.log
    def _get_info(self, d_table_name_2_ids: dict, log_id=None) -> dict:
        """ 获取具体的结构化信息 """
        d_table_id_2_info = {}

        pool = []
        for table_name, ids in d_table_name_2_ids.items():
            thread = threading.Thread(target=self._get_info_thread, args=(ids, table_name, d_table_id_2_info))
            thread.start()
            pool.append(thread)

        for thread in pool:
            thread.join()

        return d_table_id_2_info


def process_score(score) -> float:
    """ 格式化 similarity score """
    return max(min(round(float(score), 4), 1.), 0)


def combine_avg_score(avg_score, score):
    """ 结合 moving avg similarity score 与 当前 similarity score """
    score = process_score(score)
    if avg_score < 0.7:
        return score * 0.85 + 0.15 * avg_score
    else:
        return score


def get_nlist(count: int):
    if count <= 80:
        return 1
    elif count <= 300:
        return int(math.sqrt(count) / 2)
    elif count <= 1000:
        return int(math.sqrt(count) * 0.7)
    elif count <= 5000:
        return int(math.sqrt(count) * 0.9)
    elif count <= 15000:
        return int(math.sqrt(count) * 1.2)
    elif count <= 50000:
        return int(math.sqrt(count) * 1.5)
    else:
        return min(int(math.sqrt(count) * 2), 2048)


def get_index(count: int, dim: int, nlist=None):
    if not nlist:
        nlist = get_nlist(count)

    if count <= 1024:
        return faiss.IndexFlatIP(dim)
    elif count <= 20000:
        quantizer = faiss.IndexFlatIP(dim)
        return faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    else:
        quantizer = faiss.IndexFlatIP(dim)
        return faiss.IndexIVFPQ(quantizer, dim, nlist, int(dim / 4), 8, faiss.METRIC_INNER_PRODUCT)


def get_table_name(tenant: str, index_name: str, partition: str = ''):
    return f'{tenant}__{index_name}__{partition}'


def get_metric(metric_type: int):
    if metric_type == 0:
        return 'inner_product'
    elif metric_type == 1:
        return 'L1'
    elif metric_type == 2:
        return 'L2'
    elif metric_type == 3:
        return 'L_inf'
    elif metric_type == 4:
        return 'Lp'
    elif metric_type == 22:
        return 'JensenShannon'
    else:
        return ''


def get_index_type(index: faiss.Index) -> str:
    for index_type, index_class in {
        'FlatIP': faiss.IndexFlatIP,
        'IVFFlat': faiss.IndexIVFFlat,
        'IVFPQ': faiss.IndexIVFPQ,
        'Flat': faiss.IndexFlat,
        'IVF': faiss.IndexIVF,
        'index': faiss.Index,
    }.items():
        if isinstance(index, index_class):
            return index_type
    return ''


@logs.log
def _combine_results(results: List[list], avg_results: List[dict], d_table_id_2_info: dict, top_k: int, log_id=None):
    for _i, one_results in enumerate(results):
        new_result = []

        tmp_avg_result = avg_results[_i]
        for val in one_results:
            table_id = f"{val['table_name']}__{val['id']}"
            data = d_table_id_2_info[table_id] if table_id in d_table_id_2_info else None
            if not data:
                continue

            if val['table_name'] not in tmp_avg_result:
                avg_similarity = 1.
            else:
                _partition = data['field'] if 'field' in data else ''
                tmp_avg_ret = tmp_avg_result[val['table_name']]
                avg_similarity = tmp_avg_ret[_partition] if _partition in tmp_avg_ret else 0.

            new_result.append({
                'data': data,
                'score': combine_avg_score(avg_similarity, val['score']),
                'mv_score': float(avg_similarity),
            })

        new_result.sort(key=lambda x: (-x['score'], -x['mv_score']))

        d_new_result = {}
        for v in new_result:
            k = f'{v}'
            if k not in d_new_result:
                d_new_result[k] = v
            if len(d_new_result) >= top_k:
                break

        results[_i] = list(d_new_result.values())

    return results


o_faiss = Faiss()
