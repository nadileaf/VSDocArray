import os
import math
import faiss
import threading
import numpy as np
import pandas as pd
from typing import List, Union
from six.moves import cPickle as pickle
from sklearn.metrics.pairwise import cosine_similarity
from vsDa.config.path import INDEX_DIR, SQLITE_DIR
from vsDa.lib.utils import md5, get_relative_file
from vsDa.lib import logs


class Faiss:
    DEFAULT = '__default'

    def __init__(self, data_dir: str = None, index_dir: str = None):
        self.data_dir = data_dir if data_dir and os.path.isdir(data_dir) else SQLITE_DIR
        self.idx_dir = index_dir if index_dir and os.path.isdir(index_dir) else INDEX_DIR

        # 记录索引
        self.indices = {}
        self.df = {}

        # 记录 index 每个分区的滑动平均向量
        self.mv_indices = {}

        print(f'DATA_ROOT: {self.data_dir}')
        print(f'INDEX_ROOT: {self.idx_dir}')

    def get_df_path(self, tenant: str, index_name: str, partition: str = ''):
        partition = partition if partition != self.DEFAULT else ''
        table_name = get_table_name(tenant, index_name, partition)
        file_name = f'{table_name}.csv'
        return os.path.join(self.data_dir, file_name)

    @logs.log
    def get_df(self, tenant: str, index_name: str, partition: str = '',
               skip_rows=None, names=None, chunk_size=None, refresh=False, log_id=None):
        file_path = self.get_df_path(tenant, index_name, partition)

        if not refresh and file_path in self.df and not skip_rows and not chunk_size:
            return self.df[file_path]

        if not os.path.exists(file_path):
            return pd.DataFrame()

        if skip_rows and not names:
            tmp_gen = pd.read_csv(file_path, chunksize=1)
            tmp_df = next(tmp_gen)
            names = list(tmp_df.columns)

        if chunk_size is None:
            df = pd.read_csv(file_path, skiprows=skip_rows, names=names)

            if skip_rows is None:
                self.df[file_path] = df
            return df

        else:
            df_gen = pd.read_csv(file_path, skiprows=skip_rows, names=names, chunksize=chunk_size)
            return next(df_gen)

    def release_df(self, tenant: str, index_name: str, partition: str = ''):
        file_path = self.get_df_path(tenant, index_name, partition)
        if file_path in self.df:
            del self.df[file_path]

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
        if not texts and not info:
            return None, [], []

        if info:
            for i, val in enumerate(info):
                if 'text' not in val:
                    val['text'] = texts[i]
                val['alias'] = texts[i]

                if 'mid' in val:
                    del val['mid']
                val['mid'] = md5(val)
        else:
            info = list(map(lambda x: {'text': x, 'alias': x, 'mid': md5({'text': x, 'alias': x})}, texts))

        names = list(info[0].keys())

        filter_indices = []
        in_df_indices = []

        d_mid = {}
        for i, v in enumerate(info):
            mid = v['mid']
            if mid not in d_mid:
                d_mid[mid] = True
            else:
                filter_indices.append(i)

        offset = 0

        file_path = self.get_df_path(tenant, index_name, partition)
        exist_file = os.path.exists(file_path)

        if exist_file:
            df_gen = pd.read_csv(file_path, names=names, chunksize=500000)
            for df in df_gen:
                len_df = len(df)
                if not len_df:
                    continue

                offset += len_df

                if 'mid' in df:
                    d_text = dict(zip(df['mid'], df.index))
                    for i, val in enumerate(info):
                        mid = val['mid']
                        if mid not in d_text:
                            filter_indices.append(i)
                        else:
                            in_df_indices.append(d_text[mid])

                else:
                    d_text = dict(zip(df['alias'], df.index))
                    for i, val in enumerate(info):
                        _text = val['alias']
                        if _text not in d_text:
                            filter_indices.append(i)
                        else:
                            in_df_indices.append(d_text[_text])

            if not filter_indices:
                if not filter_exist:
                    mid_list = list(map(lambda x: x['mid'], info))
                    return np.array(in_df_indices, dtype=np.int32), list(range(len(info))), mid_list
                return None, [], []

        filter_indices = list(set(filter_indices))
        filter_indices.sort()

        filter_info = [info[i] for i in filter_indices] if filter_indices else info
        filter_indices = list(range(len(info))) if not filter_indices else filter_indices

        df = pd.DataFrame(filter_info, columns=names)
        df.to_csv(file_path, index=False, mode='a', header=False if exist_file else True)

        mid_list = list(map(lambda x: x['mid'], filter_info))
        return np.arange(offset, offset + len(filter_indices)), filter_indices, mid_list

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

    def list_info(self, tenant: str, index_name: str, partition: str = '', names=None, log_id=None) -> list:
        df = self.get_df(tenant, index_name, partition, names=names)
        len_df = len(df)
        if not len_df:
            return []

        names = list(df.columns) if not names else names

        _info = [{n: df[n][_i] for n in names} for _i in range(len_df)]
        if 'mid' in df:
            _info = list({v['mid']: v for v in _info}.items())
        else:
            _info = list({v['text']: v for v in _info}.items())

        return _info

    @logs.log
    def save_one(self, tenant: str, index_name: str, partition: str = '', mv=False, log_id=None) -> int:
        _index = self.index(tenant, index_name, partition)
        if _index is None:
            return 0

        if not partition:
            for partition, _index in self.indices[tenant][index_name].items():
                if _index is None:
                    continue

                index_path = get_relative_file(tenant, index_name, f'{partition}.index', root=self.idx_dir)
                faiss.write_index(_index, index_path)

            if tenant in self.mv_indices and index_name in self.mv_indices[tenant] and \
                    self.mv_indices[tenant][index_name] is not None:
                with open(get_relative_file(tenant, index_name, 'mv_index.pkl', root=self.idx_dir), 'wb') as f:
                    pickle.dump(self.mv_indices[tenant][index_name], f)

        else:
            index_path = get_relative_file(tenant, index_name, f'{partition}.index', root=self.idx_dir)
            faiss.write_index(_index, index_path)

            if (partition == self.DEFAULT or mv) and tenant in self.mv_indices and \
                    index_name in self.mv_indices[tenant] and self.mv_indices[tenant][index_name] is not None:
                with open(get_relative_file(tenant, index_name, 'mv_index.pkl', root=self.idx_dir), 'wb') as f:
                    pickle.dump(self.mv_indices[tenant][index_name], f)

        return 1

    def save(self, tenant: str, log_id=None):
        """ 保存当前的所有索引到文件里 """
        if tenant not in self.indices:
            logs.add(log_id, 'save', f'tenant "{tenant}" 没有索引')
            return

        for index_name, index in self.indices[tenant].items():
            self.save_one(tenant, index_name, log_id=log_id)

    @logs.log
    def load_one(self, tenant: str, index_name: str, partition: str = '', mv=False, refresh=False, log_id=None) -> int:
        index_dir = os.path.join(self.idx_dir, tenant, index_name)
        mv_index_path = os.path.join(index_dir, 'mv_index.pkl')

        if not partition:
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
                if not refresh and self.index(tenant, index_name, partition) is not None:
                    continue

                index_path = os.path.join(index_dir, file_name)
                self.indices[tenant][index_name][partition] = faiss.read_index(index_path)

            # 若文件存在 且 没有被加载到内存
            if os.path.exists(mv_index_path) and (refresh or index_name not in self.mv_indices[tenant] or
                                                  self.mv_indices[tenant][index_name] is None):
                with open(mv_index_path, 'rb') as f:
                    self.mv_indices[tenant][index_name] = pickle.load(f)

        else:
            index_path = os.path.join(index_dir, f'{partition}.index')
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

            if refresh or self.index(tenant, index_name, partition) is None:
                self.indices[tenant][index_name][partition] = faiss.read_index(index_path)

            # 若文件存在 且 没有被加载到内存
            if mv and os.path.exists(mv_index_path):
                if tenant not in self.mv_indices:
                    self.mv_indices[tenant] = {}
                if refresh or index_name not in self.mv_indices[tenant] or self.mv_indices[tenant][index_name] is None:
                    with open(mv_index_path, 'rb') as f:
                        self.mv_indices[tenant][index_name] = pickle.load(f)

        return 1

    @logs.log
    def load_batch(self, tenants: List[str], index_names: List[str], partitions: List[str], refresh=False, log_id=None):
        pool = []
        for i, _tenant in enumerate(tenants):
            _thread = threading.Thread(target=self.load_one,
                                       args=(_tenant, index_names[i], partitions[i]),
                                       kwargs={'refresh': refresh, 'log_id': f'{log_id}({i})'})
            _thread.start()
            pool.append(_thread)

        for _thread in pool:
            _thread.join()

        return 1

    @logs.log
    def load(self, tenant: str, refresh=False, log_id=None):
        """ 从文件中加载索引 """
        _tenant_dir = os.path.join(self.idx_dir, tenant)
        if not os.path.isdir(_tenant_dir):
            return

        for index_name in os.listdir(_tenant_dir):
            self.load_one(tenant, index_name, refresh=refresh, log_id=log_id)

    def release(self, tenant: str, index_name: str, partition: str = '', save=True, mv=False, log_id=None) -> int:
        # release index 前，先保存索引
        if save:
            ret = self.save_one(tenant, index_name, partition, mv=mv, log_id=log_id)
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
        if os.path.exists(os.path.join(self.idx_dir, tenant, index_name, f'{partition}.index')):
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
               columns: List[str] = None,
               nprobe=10,
               top_k=20,
               each_top_k=20,
               use_mv=True,
               threshold: float = None,
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
                                 avg_results, use_mv, d_table_name_2_ids, results, threshold, log_id=log_id)

        # 获取具体的结构化信息
        d_table_id_2_info = self._get_info(d_table_name_2_ids, columns, log_id=log_id)

        return _combine_results(results, avg_results, d_table_id_2_info, top_k, log_id=log_id)

    def delete_with_id(self, mids: List[int], tenant: str, index_name: str, partition: str = '', log_id=None):
        if not mids:
            return

        df = self.get_df(tenant, index_name, partition)
        if len(df) == 0:
            return

        del_indices = []
        for mid in mids:
            del_indices += list(df[df['mid'] == mid].index)

        del_indices = list(set(del_indices))

        df = df.drop(del_indices)

        df_path = self.get_df_path(tenant, index_name, partition)
        df.to_csv(df_path, index=False)

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
                        threshold: float = None,
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

        m_valid = I != -1
        if threshold:
            m_valid *= D > threshold
            I[m_valid == False] = -1
        D *= m_valid

        ids = list(set(list(map(int, I.reshape(-1)))))
        if -1 in ids:
            ids.remove(-1)

        if not ids:
            return

        if table_name not in d_table_name_2_ids:
            d_table_name_2_ids[table_name] = []
        d_table_name_2_ids[table_name] += ids

        for _i, _result_ids in enumerate(I):
            similarities = D[_i]
            results[_i] += [
                {'id': _id, 'score': _similarity, 'table_name': table_name, 'index': index_name, 'partition': partition}
                for _id, _similarity in zip(_result_ids, similarities) if _id != -1
            ]

    @logs.log
    def _get_info_thread(self,
                         tenant: str,
                         index_name: str,
                         partition: str,
                         table_name: str,
                         ids: list,
                         d_table_id_2_info: dict,
                         columns: List[str] = None,
                         log_id=None):
        if not ids:
            return

        min_i = min(ids)
        max_i = max(ids)

        df = self.get_df(tenant, index_name, partition, skip_rows=min_i, chunk_size=max_i - min_i + 1, names=columns)
        for _id in ids:
            info = df.iloc[_id - min_i].to_dict()
            table_id = f"{table_name}__{_id}"
            d_table_id_2_info[table_id] = info

    @logs.log
    def _get_info(self, d_table_name_2_ids: dict, columns: List[str] = None, log_id=None) -> dict:
        d_table_id_2_info = {}
        for table_name, ids in d_table_name_2_ids.items():
            if not ids:
                continue

            clean_table_name = table_name[:-9] if table_name.endswith('__default') else table_name
            tenant, index_name, partition = clean_table_name.split('__')

            df = self.get_df(tenant, index_name, partition, log_id=log_id)
            for _id in ids:
                info = df.iloc[int(_id)].to_dict()
                table_id = f"{table_name}__{_id}"
                d_table_id_2_info[table_id] = info

        return d_table_id_2_info

    @logs.log
    def _get_info_multi_thread(self, d_table_name_2_ids: dict, columns: List[str] = None, log_id=None) -> dict:
        """ 获取具体的结构化信息 """
        d_table_id_2_info = {}

        q = []

        for table_name, ids in d_table_name_2_ids.items():
            if not ids:
                continue

            clean_table_name = table_name[:-9] if table_name.endswith('__default') else table_name
            tenant, index_name, partition = clean_table_name.split('__')

            ids.sort()

            g_ids = []
            for _id in ids:
                if not g_ids:
                    g_ids.append([_id])
                    continue

                find_g = False
                for g in g_ids:
                    max_g = max(g + [_id])
                    min_g = min(g + [_id])
                    if max_g - min_g <= 10000:
                        g.append(_id)
                        find_g = True
                        break

                if not find_g:
                    g_ids.append([_id])

            for tmp_ids in g_ids:
                q.append([tenant, index_name, partition, table_name, tmp_ids])

        pool = []
        t_i = 0
        for tenant, index_name, partition, table_name, ids in q:
            thread = threading.Thread(target=self._get_info_thread, args=(
                tenant, index_name, partition, table_name, ids, d_table_id_2_info, columns
            ), kwargs={'log_id': f'{log_id}({t_i})'})
            t_i += 1

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
                'index': val['index'],
                'partition': val['partition'],
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
