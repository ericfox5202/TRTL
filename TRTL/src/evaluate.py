import torch
from knowledgeGraph import KnowledgeGraph, TestDataset
from timeAttentionModel import TimeAttentionModel, TimeAttentionModelTrainer
import json
import os


def initialize_knowledge_graph(data_name, size_rate, dropout_rate, epc, query_r=None, MAX_L=3):
    # Define dataset configurations
    configs = {
        'yago': {
            'path': '../data/YAGO11k',
            'name': 'yago',
            'file_name_template': 'yago_rule_final_len{}_size{}_epc{}_query{}.json',
            'file_path_template': '../data/YAGO11k/rule/{}'
        },
        'wiki': {
            'path': '../data/WIKIDATA12k',
            'name': 'wiki',
            'file_name_template': 'wiki_rule_final_len{}_size{}_epc{}_query{}.json',
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
    file_name = config['file_name_template'].format(MAX_L, size_rate, epc, query_r)
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


def evaluate(data_name, query_r, size_rate=0.05, epc=1, margin=0.5, MAX_L=3, TopK=500):
    knowledge_graph, rule_file_path = initialize_knowledge_graph(data_name, size_rate, dropout_rate=-1, epc=epc, query_r=query_r, MAX_L=MAX_L)
    knowledge_graph.train_mode('test')

    with open(rule_file_path, 'r') as file:
        rules = json.load(file)
        rule_list_dict = build_rule_list_dict(rules, query_r)

    knowledge_graph.rule_list = rule_list_dict[query_r][1:]


    # === 加载测试数据集 ===
    test_dataset = TestDataset(knowledge_graph, query_r)

    # === 初始化空模型 ===
    time_attention_model = TimeAttentionModel(knowledge_graph, query_r, MAX_L, TopK)

    checkpoint_path = f'model/{data_name}/sizeRate{size_rate}_epc{epc}_query{query_r}_margin{margin}'
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"[Eero] Checkpoint not found for query {query_r}: {checkpoint_path}")
    print(f"[Loading] {checkpoint_path}")
    state_dict = torch.load(checkpoint_path)
    time_attention_model.load_state_dict(state_dict)

    # === 开始评估 ===
    time_attention_model.eval()
    trainer = TimeAttentionModelTrainer(knowledge_graph, time_attention_model)
    results = trainer.evaluate(test_dataset)
    del time_attention_model
    torch.cuda.empty_cache()
    return results
