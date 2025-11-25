import collections
import json
import os
import random
from statistics import mean, median, stdev
from typing import Tuple, Dict
from torch.utils.data import Dataset
import torch
import copy
from attentionGraph import AttentionGraph
from collections import deque
import random


# 关系编号（与原实现保持 0..12 的含义一致）
ALLEN_EQ = 0  # equal
ALLEN_BEFORE = 1  # before
ALLEN_AFTER = 2  # after
ALLEN_STARTS = 3  # starts
ALLEN_STARTED_BY = 4  # started-by
ALLEN_FINISHES = 5  # finishes
ALLEN_FINISHED_BY = 6  # finished-by
ALLEN_CONTAINS = 7  # contains
ALLEN_DURING = 8  # during
ALLEN_OVERLAPS = 9  # overlaps
ALLEN_OVERLAPPED_BY = 10  # overlapped-by
ALLEN_MEETS = 11  # meets
ALLEN_MET_BY = 12  # met-by
ALLEN_UNKNOWN = 13  # 未知/不适用（新增）

ALLEN_BUCKET = 14  # 每个关系的步长：13 个 Allen 原子 + 1 个未知槽


# ============================================================
# KnowledgeGraph：并行索引与批量候选
# ============================================================
class KnowledgeGraph(object):
    def __init__(self, data_path, name='wiki', reverse=True, size_rate=1.0, dropout_rate=-1, epc=1):
        self.dropout_rate = dropout_rate
        self.data_path = data_path
        if name == 'wiki':
            self.relation_R = 24
        elif name == 'yago':
            self.relation_R = 10

        self.entity2id = dict()
        self.relation2id = dict()
        self.id2entity = dict()
        self.id2relation = dict()

        with open(os.path.join(data_path, 'entities.dict'), encoding='utf-8') as fi:
            for line in fi:
                id_, entity = line.strip().split('\t')
                self.entity2id[entity] = int(id_)
                self.id2entity[int(id_)] = entity

        with open(os.path.join(data_path, 'relations.dict'), encoding='utf-8') as fi:
            for line in fi:
                id_, relation = line.strip().split('\t')
                self.relation2id[relation] = int(id_)
                self.id2relation[int(id_)] = relation

        self.entity_size = len(self.entity2id)
        self.relation_size = len(self.relation2id) + self.relation_R

        self.bg_facts = self.parse_facts('bg.txt', hash_fact_index=True)
        self.train_facts = self.parse_facts(f'train_{size_rate}.txt', hash_fact_index=True)
        self.valid_facts = self.parse_facts('valid.txt')
        self.test_facts  = self.parse_facts('test.txt')

        self.rule_list = None
        self.static_train_facts = self.parse_static_facts(f'train_{size_rate}.txt')
        self.static_hr2t = dict()
        self.add_to_static_hr2t(self.static_train_facts, self.static_hr2t)
        static_train_facts_rev = set()
        for h, r, t in self.static_train_facts:
            static_train_facts_rev.add((t, r + self.relation_R, h))
        self.static_train_facts |= static_train_facts_rev

        self.hr2t_train = dict()
        self.add_to_hr2t(self.train_facts, self.hr2t_train)

        self.hr2t_valid = dict()
        self.add_to_hr2t(self.bg_facts, self.hr2t_valid)

        self.hr2t_test = dict()
        self.add_to_hr2t(self.bg_facts, self.hr2t_test)
        self.add_to_hr2t(self.valid_facts, self.hr2t_test)

        self.hr2t_known = dict()
        self.add_to_hr2t(self.bg_facts, self.hr2t_known)
        self.add_to_hr2t(self.valid_facts, self.hr2t_known)
        self.add_to_hr2t(self.test_facts, self.hr2t_known)



        if reverse is True:
            self.train_facts = self.add_reverse_facts(self.train_facts, self.relation_R)
            self.test_facts = self.add_reverse_facts(self.test_facts, self.relation_R)
            self.bg_facts = self.add_reverse_facts(self.bg_facts, self.relation_R)
            self.valid_facts = self.add_reverse_facts(self.valid_facts, self.relation_R)


        self.mode = None
        self.hr2t = None
        print("Data loading | DONE!")

    def add_reverse_facts(self, facts, relation_R):
        """
        为给定 facts 列表添加逆关系 facts。
        参数:
            facts: List[(h, r, t, date, fact_index)]
            relation_R: int, 正向关系数量（用于生成 r + relation_R）
        返回:
            新列表，包含原 facts + 逆向 facts
        """
        reverse_facts = []
        for h, r, t, date, fact_index in facts:
            if r < relation_R:  # 只为正向关系加逆
                reverse_facts.append((t, r + relation_R, h, date, fact_index))
        return facts + reverse_facts

    def train_mode(self, mode='train'):
        self.mode = mode
        if mode == 'train':
            self.hr2t = self.hr2t_train
        elif mode == 'vaild':
            self.hr2t = self.hr2t_valid
        elif mode == 'test':
            self.hr2t = self.hr2t_test
        else:
            raise RuntimeError('erro mode')

    def add_to_hr2t(self, facts, hr2t_dict):
        for h, r, t, date, fact_index in facts:
            code = self.encode_hr(h, r)
            if code not in hr2t_dict:
                hr2t_dict[code] = []
            hr2t_dict[code].append((t, date))
            code_r = self.encode_hr(t, r + self.relation_R)
            if code_r not in hr2t_dict:
                hr2t_dict[code_r] = []
            hr2t_dict[code_r].append((h, date))

    def add_to_static_hr2t(self, facts, static_hr2t_dict_set):
        for h, r, t in facts:
            code = self.encode_hr(h, r)
            if code not in static_hr2t_dict_set:
                static_hr2t_dict_set[code] = set()
            static_hr2t_dict_set[code].add(t)
            code_r = self.encode_hr(t, r + self.relation_R)
            if code_r not in static_hr2t_dict_set:
                static_hr2t_dict_set[code_r] = set()
            static_hr2t_dict_set[code_r].add(h)

    def parse_facts(self, dataset_name, hash_fact_index=False):
        facts = []
        with open(os.path.join(self.data_path, dataset_name)) as file:
            for fact_index, line in enumerate(file):
                items = line.strip().split('\t')
                h, r, t, date_s, date_e = items
                date_s, date_e = int(date_s), int(date_e)
                h, r, t = int(h), int(r), int(t)
                if hash_fact_index:
                    facts.append((h, r, t, (date_s, date_e), fact_index))
                else:
                    facts.append((h, r, t, (date_s, date_e), None))
        return facts

    def parse_static_facts(self, dataset_name):
        facts = set()
        with open(os.path.join(self.data_path, dataset_name)) as file:
            for fact_index, line in enumerate(file):
                items = line.strip().split('\t')
                h, r, t, date_s, date_e = items
                h, r, t = int(h), int(r), int(t)
                facts.add((h, r, t))
        return facts

    def encode_hr(self, h, r):
        return r * self.entity_size + h

    def decode_hr(self, index):
        h = index % self.entity_size
        r = index // self.entity_size
        return h, r

    # Allen
    def get_allen(self, date1, date2):
        s1, e1 = date1
        s2, e2 = date2
        if s1 > e1 or s2 > e2:
            return None
        if s1 == s2 and e1 == e2:
            return ALLEN_EQ
        if e1 < s2:
            return ALLEN_BEFORE
        if s1 > e2:
            return ALLEN_AFTER
        if e1 == s2:
            return ALLEN_MEETS
        if s1 == e2:
            return ALLEN_MET_BY
        if s1 == s2:
            return ALLEN_STARTS if e1 < e2 else ALLEN_STARTED_BY
        if e1 == e2:
            return ALLEN_FINISHES if s1 > s2 else ALLEN_FINISHED_BY
        if s1 < s2 and e1 > e2:
            return ALLEN_CONTAINS
        if s1 > s2 and e1 < e2:
            return ALLEN_DURING
        if s1 < s2 < e1 < e2:
            return ALLEN_OVERLAPS
        if s2 < s1 < e2 < e1:
            return ALLEN_OVERLAPPED_BY
        return None

    def get_allen_atom(self, date1, date2, r):
        allen = self.get_allen(date1, date2)
        idx = allen if allen is not None else ALLEN_UNKNOWN
        return r * ALLEN_BUCKET + idx

    def get_sorted_r_list(self):
        nums = collections.defaultdict(int)
        for fact in self.train_facts:
            h, r, t, date, _ = fact
            nums[r] += 1
        r_list = [item[0] for item in sorted(nums.items(), key=lambda x: x[1], reverse=True)]
        return r_list

    # ========== 并行检索 CSR 索引 ==========
    def _select_facts_for_mode(self):
        if self.mode == 'train':
            return self.train_facts
        elif self.mode == 'vaild':
            return self.bg_facts
        elif self.mode == 'test':
            return self.bg_facts + self.valid_facts
        else:
            raise RuntimeError('error mode for parallel index')

    def enable_parallel_index(self, device: str = "cuda"):
        facts = self._select_facts_for_mode()
        E = self.entity_size
        R = self.relation_size
        HR = E * R

        counts = [0] * HR
        for h, r, t, date, _ in facts:
            counts[r * E + h] += 1

        hr_ptr = [0] * (HR + 1)
        csum = 0
        for i in range(HR):
            hr_ptr[i] = csum
            csum += counts[i]
        hr_ptr[HR] = csum
        M = csum

        hr_t = [0] * M
        hr_s = [0] * M
        hr_e = [0] * M
        write_pos = hr_ptr[:-1].copy()

        for h, r, t, (ds, de), _ in facts:
            code = r * E + h
            pos = write_pos[code]
            hr_t[pos] = t
            hr_s[pos] = ds
            hr_e[pos] = de
            write_pos[code] += 1

        self.hr_ptr = torch.tensor(hr_ptr, dtype=torch.long, device=device)
        self.hr_t   = torch.tensor(hr_t,   dtype=torch.long, device=device)
        self.hr_s   = torch.tensor(hr_s,   dtype=torch.long, device=device)
        self.hr_e   = torch.tensor(hr_e,   dtype=torch.long, device=device)

        self._HR_codes = HR
        self._E_size = E
        self._R_size = R

    @torch.no_grad()
    def enumerate_candidates_batch(
        self,
        deep: int,
        src_h: torch.Tensor,              # [K]
        src_date: torch.Tensor,           # [K,2]
        head_relation: int,               # 供 allen_head 参考
        predict_fact,                     # (h, r, t, predict_date)
        predict_fact_rev,                 # (t, r', h, predict_date)
    ) -> Dict[str, torch.Tensor]:
        device = src_h.device
        K = src_h.numel()
        if not hasattr(self, "hr_ptr"):
            self.enable_parallel_index(device=device)

        E = self._E_size
        R = self._R_size

        if self.rule_list is None:
            expand_r_list = list(range(R))
        else:
            expand_r_list = self.rule_list[deep]

        _, _, _, predict_date = predict_fact
        pf_h, pf_r, pf_t, pf_date = predict_fact
        pr_h, pr_r, pr_t, pr_date = predict_fact_rev

        cand_ptr = [0]
        v_h_list, v_s_list, v_e_list = [], [], []
        r_list, allen_list, allen_head_list = [], [], []
        total = 0

        for i in range(K):
            h_i = int(src_h[i].item())
            ds_i = int(src_date[i, 0].item())
            de_i = int(src_date[i, 1].item())
            date_i = (ds_i, de_i)

            for r in expand_r_list:
                code = r * E + h_i
                beg = int(self.hr_ptr[code].item())
                end = int(self.hr_ptr[code + 1].item())
                if beg == end:
                    continue

                t_seg = self.hr_t[beg:end]
                s_seg = self.hr_s[beg:end]
                e_seg = self.hr_e[beg:end]

                for j in range(end - beg):
                    t_j = int(t_seg[j].item())
                    ds_j = int(s_seg[j].item())
                    de_j = int(e_seg[j].item())
                    date_j = (ds_j, de_j)

                    if (h_i == pf_h and r == pf_r and t_j == pf_t and date_j == pf_date):
                        continue
                    if (h_i == pr_h and r == pr_r and t_j == pr_t and date_j == pr_date):
                        continue

                    allen_ij = self.get_allen_atom(date_i, date_j, r)
                    allen_head_ij = self.get_allen_atom(predict_date, date_j, r)

                    v_h_list.append(t_j)
                    v_s_list.append(ds_j)
                    v_e_list.append(de_j)
                    r_list.append(r)
                    allen_list.append(allen_ij)
                    allen_head_list.append(allen_head_ij)
                    total += 1

            cand_ptr.append(total)

        if total == 0:
            return {
                "cand_ptr": torch.zeros(K + 1, dtype=torch.long, device=device),
                "cand_v_h": torch.empty(0, dtype=torch.long, device=device),
                "cand_v_date": torch.empty(0, 2, dtype=torch.long, device=device),
                "cand_r": torch.empty(0, dtype=torch.long, device=device),
                "cand_allen": torch.empty(0, dtype=torch.long, device=device),
                "cand_allen_head": torch.empty(0, dtype=torch.long, device=device),
            }

        cand_ptr = torch.tensor(cand_ptr, dtype=torch.long, device=device)  # [K+1]
        cand_v_h = torch.tensor(v_h_list, dtype=torch.long, device=device)  # [M]
        cand_v_date = torch.stack([
            torch.tensor(v_s_list, dtype=torch.long, device=device),
            torch.tensor(v_e_list, dtype=torch.long, device=device)
        ], dim=1)                                                           # [M,2]
        cand_r = torch.tensor(r_list, dtype=torch.long, device=device)      # [M]
        cand_allen = torch.tensor(allen_list, dtype=torch.long, device=device)
        cand_allen_head = torch.tensor(allen_head_list, dtype=torch.long, device=device)

        return {
            "cand_ptr": cand_ptr,
            "cand_v_h": cand_v_h,
            "cand_v_date": cand_v_date,
            "cand_r": cand_r,
            "cand_allen": cand_allen,
            "cand_allen_head": cand_allen_head,
        }

    # def get_allen(self, date1, date2):
    #     start1, end1 = date1
    #     start2, end2 = date2
    #
    #     if end1 < start2:
    #         return 1
    #     elif start1 > end2:
    #         return 2
    #
    #     return 3


class TrainDataset(Dataset):
    def __init__(self, graph, query_r=None):
        self.graph = graph
        facts = copy.deepcopy(self.graph.train_facts)
        random.shuffle(facts)

        self.batches = []
        for h, r, t, date, fact_index in facts:
            if r == query_r:
                self.batches.append([h, r, t, date, fact_index])

    def __len__(self):
        return len(self.batches)

    def calculateOverlap(self, l1, l2):
        a1, a2 = l1
        b1, b2 = l2
        start = max(a1, b1)
        end = min(a2, b2)
        return max(0, end - start + 1)

    def __getitem__(self, idx):
        """
            target: [[命中的下标对应为1] * batch_size]
            """
        h, r, t, date, fact_index = self.batches[idx]
        fact_indexes = [fact_index]
        target = torch.zeros(self.graph.entity_size, dtype=torch.float)
        target[t] = 1

        return h, r, t, date, target, fact_indexes


class TestDataset(Dataset):
    def __init__(self, graph, query_r):
        self.graph = graph

        facts = copy.deepcopy(self.graph.test_facts)
        random.shuffle(facts)

        self.batches = []
        for h, r, t, date, fact_index in facts:
            if r == query_r:
                self.batches.append([h, r, t, date, fact_index])

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        """
        target: [[命中的下标对应为1] * batch_size]
        """
        h, r, t, date, fact_index = self.batches[idx]
        code = self.graph.encode_hr(h, r)
        flags = []
        if code not in self.graph.hr2t_known:
            flag = torch.ones(self.graph.entity_size, dtype=torch.bool)
            flag[t] = False
            flags.append(flag)
        else:
            for d in range(date[0], date[1] + 1):
                flag = torch.ones(self.graph.entity_size, dtype=torch.bool)
                flag[t] = False
                for ct, ct_date in self.graph.hr2t_known[code]:
                    if ct_date[0] <= d <= ct_date[1]:
                        flag[ct] = False
                flags.append(flag)

        return h, r, t, date, flags, [fact_index]


