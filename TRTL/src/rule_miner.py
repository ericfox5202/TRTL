import json

def rule_mine(knowledgeGraph, file_name, query_r, MAX_L=3):
    rule_set = set()
    process_idx = 0
    hit_num = 0
    for h_head, r_head, t_head in knowledgeGraph.static_train_facts:
        process_idx += 1
        if r_head != query_r:
            continue

        target_fact = (h_head, r_head, t_head)

        path = {((r_head, ), h_head)}
        hit = False
        for deep in range(MAX_L):
            path_now = set()
            for path_r, path_e in path:
                for now_r in range(knowledgeGraph.relation_size):
                    code = knowledgeGraph.encode_hr(path_e, now_r)
                    if code not in knowledgeGraph.static_hr2t:
                        continue

                    for t in knowledgeGraph.static_hr2t[code]:
                        now_path = list(path_r) + [now_r]
                        if len(now_path) > MAX_L + 1:
                            continue  # 控制路径长度

                        if t == t_head:
                            rule_set.add(tuple(now_path))
                            hit = True

                        path_now.add((tuple(now_path), t))

            path = path_now
        if hit is True:
            hit_num += 1

    rule_list = []
    for rule in rule_set:
        rule_list.append(list(rule))

    with open(file_name, 'w') as file:
        json.dump(rule_list, file)
