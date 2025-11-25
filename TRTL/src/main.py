import torch
import logging
from knowledgeGraph import KnowledgeGraph, TrainDataset, TestDataset
from timeAttentionModel import TimeAttentionModel, TimeAttentionModelTrainer
import json
import sys
import glob
import os
from rule_miner import rule_mine
from collections import defaultdict
from itertools import zip_longest
import multiprocessing
from train import train
from evaluate import evaluate
from statistics import mean, median, stdev
from collections import Counter

lr = 0.001

def train_for_query(data_name, query_r, size_rate, epc, score_margin, MAX_L, TopK):
    train(data_name=data_name, size_rate=size_rate, epc=epc,
          lr=lr, margin=score_margin, query_r=query_r, MAX_L=MAX_L, TopK=TopK)

def run_eval(data_name, query_r, size_rate, epc, score_margin, MAX_L, TopK):
    result = evaluate(data_name=data_name, query_r=query_r,
                      size_rate=size_rate, epc=epc,
                      margin=score_margin, MAX_L=MAX_L, TopK=TopK)
    print(f"[Result] Query {query_r}: {result}")
    return query_r, result


if __name__ == '__main__':
    import argparse

    # ========= 1. 解析命令行参数 =========
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataname',
        type=str,
        default='yago',
        help='dataset name, e.g. yago or wiki'
    )
    parser.add_argument(
        '--maxL',
        type=int,
        default=5,
        help='maximum query length (MAX_L)'
    )
    parser.add_argument(
        '--TopK',
        type=int,
        default=10,
        help='top-k value'
    )
    args = parser.parse_args()

    data_name = args.dataname
    MAX_L = args.maxL
    TopK = args.TopK  # 先保存下来，后面需要用的时候直接从这里拿

    # ========= 2. 根据 dataname 选择 data_path =========
    if data_name.lower() == 'yago':
        data_path = '../data/YAGO11k'
    elif data_name.lower() == 'wiki':
        data_path = '../data/WIKIDATA12k'
    else:
        # 如果你有别的数据集，可以按需改这里
        data_path = os.path.join('../data', data_name)

    ctx = multiprocessing.get_context("spawn")

    # 设置日志配置
    logging.basicConfig(
        filename='results.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    logging.info(
        f'Args: dataname={data_name}, data_path={data_path}, MAX_L={MAX_L}, topk={TopK}'
    )

    for epc in range(2, 3):
        for margin in [0.5]:
            for size_rate in [1.0]:
                logging.info('-----train start-----')
                kg = KnowledgeGraph(
                    data_path,
                    name=data_name,
                    reverse=True,
                    size_rate=size_rate,
                    dropout_rate=-1,
                    epc=epc
                )
                relation_size = kg.relation_size
                kg.train_mode('train')

                # 设置多进程
                num_processes = min(relation_size, 4)
                query_r_list = kg.get_sorted_r_list()
                query_r_not_view = set(query_r_list)

                split_idx = -1
                if split_idx > 0:
                    with ctx.Pool(processes=split_idx, maxtasksperchild=1) as pool:
                        args_list = [
                            (data_name, query_r, size_rate, epc, margin, MAX_L, TopK)
                            for query_r in query_r_list[:split_idx]
                        ]
                        pool.starmap(train_for_query, args_list)
                    with ctx.Pool(processes=split_idx, maxtasksperchild=1) as pool:
                        args_list = [
                            (data_name, query_r, size_rate, epc, margin, MAX_L, TopK)
                            for query_r in query_r_list[split_idx:]
                        ]
                        pool.starmap(train_for_query, args_list)
                else:
                    with ctx.Pool(processes=num_processes, maxtasksperchild=1) as pool:
                        args_list = [
                            (data_name, query_r, size_rate, epc, margin, MAX_L, TopK)
                            for query_r in query_r_list
                        ]
                        pool.starmap(train_for_query, args_list)

                logging.info('-----train end-----')
                logging.info('-----eval start-----')
                with ctx.Pool(processes=num_processes, maxtasksperchild=1) as pool:
                    args_list = [
                        (data_name, query_r, size_rate, epc, margin, MAX_L, TopK)
                        for query_r in query_r_list
                    ]
                    results = pool.starmap(run_eval, args_list)

                logging.info('-----eval end-----')

                # 汇总结果
                results_dict = {qid: res for qid, res in results}
                print("\nAll finished.\nSummary(size_rate_{}):".format(size_rate))

                hit1 = 0
                hit5 = 0
                hit10 = 0
                mrr = 0
                n = 0

                logging.info(
                    'dataset: {}, size_rate: {}, epc: {}, lr: {}'.format(
                        data_name, size_rate, epc, lr
                    )
                )
                for qid, metrics in results_dict.items():
                    query_r_not_view.remove(qid)
                    hit1 += metrics['hit1']
                    hit5 += metrics['hit5']
                    hit10 += metrics['hit10']
                    mrr += metrics['mrr']
                    n += metrics['data_num']
                    logging.info(
                        'query: {}, hit1: {}, hit5: {}, hit10: {}, raw_mrr: {}'.format(
                            qid, metrics['hit1'], metrics['hit5'],
                            metrics['hit10'], metrics['mrr']
                        )
                    )

                logging.info(
                    'dataset: {}, size_rate: {}, epc: {}, lr: {}'.format(
                        data_name, size_rate, epc, lr
                    )
                )
                if len(query_r_not_view) == 0:
                    logging.info('all finished')
                else:
                    logging.info('not finished')
                logging.info(
                    'hit1: {}, hit5: {}, hit10: {}, mrr: {}'.format(
                        hit1 / n, hit5 / n, hit10 / n, mrr / n
                    )
                )
