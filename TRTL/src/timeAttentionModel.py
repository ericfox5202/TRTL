import torch
from torch import nn
from datetime import datetime
from attentionGraph import AttentionGraph

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 段/CSR 工具
# ============================================================
def _segment_max(x: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    assert x.dim() == 1 and index.dim() == 1
    device = x.device
    out = torch.full((dim_size,), float("-inf"), device=device, dtype=x.dtype)
    # torch>=2.0 支持
    out = out.scatter_reduce(0, index, x, reduce='amax', include_self=True)
    return out

def _segment_sum(x: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    assert x.dim() == 1 and index.dim() == 1
    device = x.device
    out = torch.zeros((dim_size,), device=device, dtype=x.dtype)
    out.index_add_(0, index, x)
    return out

def segment_softmax(x: torch.Tensor, index: torch.Tensor, dim_size: int, eps: float = 1e-9) -> torch.Tensor:
    seg_max = _segment_max(x, index, dim_size)                 # [dim_size]
    x_exp = torch.exp(x - seg_max[index])                      # [L]
    seg_sum = _segment_sum(x_exp, index, dim_size) + eps       # [dim_size]
    return x_exp / seg_sum[index]                              # [L]

def segment_scatter_add(x: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    device = x.device
    out = torch.zeros((dim_size,), device=device, dtype=x.dtype)
    out.index_add_(0, index, x)
    return out

def segment_scatter_add_mat(x: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    device = x.device
    H = x.size(1)
    out = torch.zeros((dim_size, H), device=device, dtype=x.dtype)
    out.index_add_(0, index, x)
    return out

# ============================================================
# ATL：ChildSumTreeLSTMWithAttention（段接口）
# ============================================================
class ChildSumTreeLSTMWithAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_a = nn.Linear(hidden_size, hidden_size)
        self.U_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v   = nn.Linear(hidden_size, 1, bias=False)

        self.tanh = nn.Tanh()

    @torch.no_grad()
    def segment_attention(
        self,
        x_emb: torch.Tensor,                 # [M,H]
        child_h: torch.Tensor,               # [E_prev,H]
        pair_cand_idx: torch.Tensor,         # [L]
        pair_child_idx: torch.Tensor,        # [L] —— 注意：必须是真实全局孩子边 id
    ):
        M = x_emb.size(0)
        L = pair_cand_idx.numel()
        if L == 0:
            attn = torch.empty(0, device=x_emb.device)
            h_hat = torch.zeros(M, x_emb.size(1), device=x_emb.device, dtype=x_emb.dtype)
            return attn, h_hat

        Ah = self.W_a(child_h)                                  # [E_prev,H]
        Bx = self.U_a(x_emb)                                    # [M,H]
        e = self.v(self.tanh(Ah[pair_child_idx] + Bx[pair_cand_idx])).squeeze(-1)  # [L]
        attn = segment_softmax(e, index=pair_cand_idx, dim_size=M)                 # [L]
        weighted = attn.unsqueeze(1) * child_h[pair_child_idx]                     # [L,H]
        h_hat = segment_scatter_add_mat(weighted, pair_cand_idx, dim_size=M)       # [M,H]
        return attn, h_hat

    def segment_forward(
        self,
        x_emb: torch.Tensor,                 # [M,H]
        child_h: torch.Tensor,               # [E_prev,H]
        child_c: torch.Tensor,               # [E_prev,H]
        h_hat: torch.Tensor,                 # [M,H]
        pair_cand_idx: torch.Tensor,         # [L]
        pair_child_idx: torch.Tensor,        # [L]
    ):
        M = x_emb.size(0)
        if pair_cand_idx.numel() == 0:
            i_j = torch.sigmoid(self.W_i(x_emb) + self.U_i(h_hat))     # [M,H]
            c_t = self.tanh(self.W_c(x_emb) + self.U_c(h_hat))         # [M,H]
            c = i_j * c_t
            o_j = torch.sigmoid(self.W_o(x_emb) + self.U_o(h_hat))     # [M,H]
            h = o_j * self.tanh(c)
            return h, c

        i_j = torch.sigmoid(self.W_i(x_emb) + self.U_i(h_hat))         # [M,H]
        Wfx = self.W_f(x_emb)                                          # [M,H]
        Ufh = self.U_f(child_h)                                        # [E_prev,H]
        f_pair = torch.sigmoid(Wfx[pair_cand_idx] + Ufh[pair_child_idx])    # [L,H]
        sum_f_c = segment_scatter_add_mat(f_pair * child_c[pair_child_idx], pair_cand_idx, dim_size=M)  # [M,H]
        c_t = self.tanh(self.W_c(x_emb) + self.U_c(h_hat))             # [M,H]
        c = i_j * c_t + sum_f_c
        o_j = torch.sigmoid(self.W_o(x_emb) + self.U_o(h_hat))         # [M,H]
        h = o_j * self.tanh(c)
        return h, c

# ============================================================
# Pruner（段接口）
# ============================================================
class Pruner(nn.Module):
    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 128,
        dropout: float = 0.1,
        use_adapter: bool = True,
    ):
        super().__init__()
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapter = nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, proj_dim),
            )
            hidden_dim = proj_dim
        else:
            hidden_dim = in_dim
        self.scorer = nn.Linear(hidden_dim, 1)

    @torch.no_grad()
    def _normalize_prev_prob(
        self,
        prev_p: torch.Tensor,           # [E_prev]
        child_src_node: torch.Tensor,   # [E_prev]（每条孩子边属于哪个源节点）
        tau_prev: float = 1.0,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if prev_p.numel() == 0:
            return prev_p
        K = int(child_src_node.max().item()) + 1 if child_src_node.numel() > 0 else 0
        z = prev_p / max(tau_prev, eps)
        q = segment_softmax(z, index=child_src_node, dim_size=K, eps=eps)  # [E_prev]
        return q.clamp(eps, 1.0 - eps)

    def segment_forward(
        self,
        x_pruner: torch.Tensor,         # [M,in_dim] = cat([x_emb,h_hat])
        attention_weight: torch.Tensor, # [L] or [L,1]
        pre_edge_prob: torch.Tensor,    # [E_prev]
        pair_cand_idx: torch.Tensor,    # [L]
        pair_child_idx: torch.Tensor,   # [L]  —— 真实孩子边 id
        child_src_node: torch.Tensor,   # [E_prev]
        tau_prev: float = 1.0,
        alpha: float = 1.0,
        eps: float = 1e-6,
    ):
        x_pruner = x_pruner.detach()
        M = x_pruner.size(0)
        h = self.adapter(x_pruner) if self.use_adapter else x_pruner
        logits = self.scorer(h).squeeze(-1)           # [M]
        s = torch.sigmoid(logits)                     # [M]

        q = self._normalize_prev_prob(pre_edge_prob, child_src_node, tau_prev=tau_prev, eps=eps)  # [E_prev]
        attn = attention_weight.squeeze(-1) if attention_weight.dim() == 2 else attention_weight  # [L]
        w_pair = attn * q[pair_child_idx]             # [L]
        agg_prob = segment_scatter_add(w_pair, pair_cand_idx, dim_size=M).clamp(eps, 1.0 - eps)   # [M]
        ctx_score = torch.logit(agg_prob)
        p = logits + alpha * ctx_score                # [M]
        return s, p

class AllenAttentionModel(nn.Module):
    def __init__(self, allen_size, relation_size, embedding_dim, training=True):
        super().__init__()
        self.MAX_LEN = 6

        self.allen_embedding = nn.Embedding(allen_size, embedding_dim)
        self.allen_head_embedding = nn.Embedding(allen_size, embedding_dim)
        self.relation_embedding = nn.Embedding(relation_size, embedding_dim)
        self.hidden_size = embedding_dim * 2
        self.rnn = ChildSumTreeLSTMWithAttention(embedding_dim * 2, embedding_dim * 2)

        self.head_relation_len_embedding_list = nn.ModuleList()
        self.trans_matrix_list = nn.ModuleList()
        for i in range(self.MAX_LEN):
            self.head_relation_len_embedding_list.append(nn.Embedding(relation_size, embedding_dim * 2))
            self.trans_matrix_list.append(nn.Linear(embedding_dim * 2, embedding_dim * 2))

        self.head_relation_len_weight = torch.nn.Parameter(torch.ones(self.MAX_LEN, dtype=torch.float))

        self.pruner = Pruner(embedding_dim * 4, proj_dim=embedding_dim)


class TimeAttentionModel(nn.Module):

    def __init__(self, knowledge_graph, query_r, MAX_L, TopK):
        super().__init__()
        self.knowledge_graph = knowledge_graph
        if knowledge_graph.mode == 'train':
            training = True
        else:
            training = False
        self.attention_model_dict = nn.ModuleDict()

        self.attention_model_dict[str(query_r)] = AllenAttentionModel(
                knowledge_graph.relation_size * 16,
                knowledge_graph.relation_size,
                128,
                training
            ).cuda()

        self.MAX_LEN = MAX_L
        self.TopK = TopK

    def get_attention_graph(self, query_h, query_r, query_t, query_date, TopK):
        predict_fact = (query_h, query_r, query_t, query_date)
        attention_graph = AttentionGraph(self.knowledge_graph, self.attention_model_dict[str(query_r)], predict_fact, TopK)
        return attention_graph

    def forward(self, quarry_h, quarry_r, quarry_t, quarry_date):
        score = torch.zeros(self.knowledge_graph.entity_size, dtype=torch.float).cuda()
        prob = torch.zeros(self.knowledge_graph.entity_size, dtype=torch.float).cuda()
        mask = torch.zeros_like(score).cuda()
        nums = torch.zeros(self.knowledge_graph.entity_size, dtype=torch.float).cuda()

        attention_graph = self.get_attention_graph(quarry_h, quarry_r, quarry_t, quarry_date, self.TopK)
        if attention_graph is None:
            mask = (mask != 0)
            return score, mask

        weight_list, target_index_list, prob_list = attention_graph.forward(self.MAX_LEN)
        for s, p, target_index in zip(weight_list, prob_list, target_index_list):
            mask[target_index] = 1
            score.scatter_add_(0, target_index, s)
            prob.scatter_add_(0, target_index, p)

            # nums.scatter_add_(0, target_index, torch.ones_like(target_index, dtype=torch.float))  # 计数

        # 只在 prob 非零处除以 nums（防止除以 0）
        # nonzero_mask = nums > 0
        # prob[nonzero_mask] = prob[nonzero_mask] / nums[nonzero_mask]

        mask = (mask != 0)

        return score, prob, mask

def ranking_loss(scores, labels, margin=0.5):
    relevant_scores = scores[labels > 0]
    irrelevant_scores = scores[labels == 0]

    if relevant_scores.numel() == 0:
        return torch.tensor(1.0, device=scores.device, requires_grad=True)

    loss_matrix = F.relu(irrelevant_scores.unsqueeze(1) - relevant_scores + margin)

    # 只平均非零项
    if (loss_matrix > 0).sum() > 0:
        return loss_matrix[loss_matrix > 0].mean()
    else:
        # 没有任何违反 margin 的项，返回 0
        return torch.tensor(0.0, device=scores.device, requires_grad=True)


class TimeAttentionModelTrainer:
    def __init__(self, knowledgeGraph, model):
        self.model = model
        self.knowledgeGraph = knowledgeGraph

    def train(self, data_set, margin, **optimizer_attr):
        optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_attr)
        self.model.train()

        total_score_loss = 0
        total_pruner_loss = 0
        batch_size = 100

        for ep in range(1, 2):

            print('ep: {}'.format(ep))
            for batch_id, batch in enumerate(data_set):
                h, r, t, date, target, fact_index = batch
                target = target.cuda()

                if (batch_id + 1) % batch_size == 0:
                    print('[{}] query: {}, batch: {}/{}, score_loss: {:.4f}, pruner_loss: {:.4f}'.format(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), r,
                        batch_id, len(data_set), total_score_loss / batch_size, total_pruner_loss / batch_size))
                    total_score_loss = 0
                    total_pruner_loss = 0

                score, prob, mask = self.model(h, r, t, date)
                if mask.sum().item() == 0 or mask[t] is False:
                    continue

                score_loss = ranking_loss(score[mask], target[mask], margin)
                pruner_loss = ranking_loss(prob[mask], target[mask], 0.5)

                optimizer.zero_grad()  # 清零梯度
                score_loss.backward()
                pruner_loss.backward()
                optimizer.step()  # 更新参数

                total_score_loss += score_loss.item()
                total_pruner_loss += pruner_loss.item()

    @torch.no_grad()
    def evaluate(self, dataset, expectation=True):
        model = self.model
        model.eval()

        hit = 0
        hit1 = 0
        hit5 = 0
        hit10 = 0
        mrr = 0
        hit_tmp = 0
        times = 0

        null_mrr = 0
        for rank in range(1, self.knowledgeGraph.entity_size):
            null_mrr += 1.0 / rank / self.knowledgeGraph.entity_size

        for h, r, t, date, flags, fact_idx in dataset:
            score, prob, mask = self.model(h, r, t, date)
            score[~mask] = float('-inf')
            val = score[t]

            n = len(flags)
            for flag in flags:
                l = (score[flag] > val).sum().item() + 1
                h = (score[flag] >= val).sum().item() + 2
                if mask[t]:
                    hit += 1 / n

                    for rank in range(l, h):
                        if rank == 1:
                            hit1 += (1 / (h - l)) / n
                        if rank <= 5:
                            hit5 += (1 / (h - l)) / n
                        if rank <= 10:
                            hit10 += (1 / (h - l)) / n

                        mrr += (1.0 / rank / (h - l)) / n
                else:
                    mrr += null_mrr / n

            times += 1
            if times % 100 == 0:
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Eval progress: {times}/{len(dataset)} ({(times / len(dataset)) * 100:.2f}%)")

        data_len = len(dataset)
        results = {
            "data_num": data_len,
            "hit": hit,
            "hit_tmp": hit_tmp,
            "hit1": hit1,
            "hit5": hit5,
            "hit10": hit10,
            "mrr": mrr,
        }

        return results




