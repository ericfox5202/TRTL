import torch
from knowledgeGraph import KnowledgeGraph, TrainDataset, TestDataset
from timeAttentionModel import TimeAttentionModel, TimeAttentionModelTrainer
import json
import sys
import glob
import os
from rule_miner import rule_mine
from collections import defaultdict
import logging

def initialize_knowledge_graph(data_name, size_rate, dropout_rate, epc, query_r):
    # Define dataset configurations
    configs = {
        'yago': {
            'path': '../data/YAGO11k',
            'name': 'yago',
            'file_name_template': 'yago_rule_final_len3_size{}_epc{}_query{}.json',
            'file_path_template': '../data/YAGO11k/rule/{}'
        },
        'wiki': {
            'path': '../data/WIKIDATA12k',
            'name': 'wiki',
            'file_name_template': 'wiki_rule_final_len3_size{}_epc{}_query{}.json',
            'file_path_template': '../data/WIKIDATA12k/rule/{}'
        }
    }

    if data_name not in configs:
        raise ValueError(f"Unsupported dataset: {data_name}")

    config = configs[data_name]
    # Initialize KnowledgeGraph
    knowledge_graph = KnowledgeGraph(
        config['path'],
        name=config['name'],
        reverse=True,
        size_rate=size_rate,
        dropout_rate=dropout_rate,
        epc=epc
    )

    # Format file name based on dataset
    file_name = config['file_name_template'].format(size_rate, epc, query_r)
    file_path = config['file_path_template'].format(file_name)

    return knowledge_graph, file_path


def build_rule_list_dict(rules, query_r):
    """
    根据传入的规则构建指定 query_r 的 rule_list_dict 条目。

    参数:
        rules: 所有挖掘出来的规则（从 json 文件加载）
        query_r: 当前要处理的 query 关系（只保留与该 query_r 相关的规则）

    返回:
        rule_list_dict: 用于构造模型的结构字典
    """
    rule_dict = {}
    for rule_index, rule in enumerate(rules):
        if len(rule) < 2:
            continue
        if rule[0] != query_r:
            continue
        if query_r not in rule_dict:
            rule_dict[query_r] = []
        rule_dict[query_r].append((rule, rule_index))

    rule_list_dict = {}
    if query_r not in rule_list_dict:
        rule_list_dict[query_r] = []
    for rule, _ in rule_dict.get(query_r, []):
        for idx, r in enumerate(rule):
            if idx == len(rule_list_dict[query_r]):
                rule_list_dict[query_r].append([])
            if r not in rule_list_dict[query_r][idx]:
                rule_list_dict[query_r][idx].append(r)

    return rule_list_dict

def get_graphRate(data_name, size_rate, epc=0):
    kg = KnowledgeGraph(data_path, name=data_name, reverse=True, size_rate=size_rate, dropout_rate=-1, epc=epc)
    relation_size = kg.relation_size

    total_num = 0
    hit_num = 0

    for query_r in range(relation_size):
        knowledge_graph, rule_file_path = initialize_knowledge_graph(data_name, size_rate, dropout_rate=-1, epc=epc, query_r=query_r)
        knowledge_graph.train_mode('train')

        if not os.path.exists(rule_file_path):
            rule_mine(knowledge_graph, rule_file_path, query_r)

        with open(rule_file_path, 'r') as file:
            rules = json.load(file)

        rule_list_dict = build_rule_list_dict(rules, query_r)
        train_dataset = TrainDataset(knowledge_graph, query_r)
        for batch_id, batch in enumerate(train_dataset):
            total_num += 1
            query_h, query_r_bath, query_t, query_date, target, fact_index = batch
            assert query_r == query_r_bath

            rule_list = rule_list_dict[query_r]
            attention_graph = knowledge_graph.ground(query_h, query_r, query_t, query_date, rule_list)
            if attention_graph is None:
                continue
            for deep in range(0, attention_graph.max_deep + 1):
                if (query_t == attention_graph.layer_index2main_key[deep]).any():
                    hit_num += 1
                    break

    return hit_num/total_num

if __name__ == '__main__':
    # 获取总的关系数量（用一个临时图加载）
    # train_for_query(0)
    data_path = '../data/WIKIDATA12k'
    data_name = 'wiki'
    # [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for size_rate in [1.0]:
        graphRate = get_graphRate(data_name, size_rate)
        logging.basicConfig(filename='graphRate.log',  # 日志文件名
                            filemode='a',  # 追加模式（如需每次覆盖写入可用 'w'）
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO)  # 设置日志级别为 INFO
        logging.info('dataset: {}, size_rate: {}, graphRate: {}'.format(data_name, size_rate, graphRate))
