import torch
import torch.nn.functional as F

class AttentionGraph:
    """
    并行版 AttentionGraph（修复版）
    - 与旧版数学行为对齐：
      (a) 第0层使用 M×M 的全对全注意力，使 agg_prob≈1/M；
      (b) 非第0层在构造“孩子 CSR”时同步重排 prev_* 到按弧头排序后的索引空间，保持按节点分段隔离一致。
    - 仍保留跨节点的全层 top-k（与旧版一致）。
    """

    def __init__(self, knowledge_graph, attention_model, predict_fact, TopK):
        self.TopK = int(TopK)
        self.kg = knowledge_graph
        self.attn = attention_model
        self.predict_fact = predict_fact

        predict_h, predict_r, predict_t, predict_date = predict_fact
        if predict_r >= self.kg.relation_R:
            self.predict_fact_rev = (predict_t, predict_r - self.kg.relation_R, predict_h, predict_date)
        else:
            self.predict_fact_rev = (predict_t, predict_r + self.kg.relation_R, predict_h, predict_date)

        self.head_relation = predict_r

        # 节点（每层）
        self.nodes_h   = []  # list[ LongTensor[K_d] ]
        self.nodes_date= []  # list[ LongTensor[K_d,2] ]

        # 边（每层）
        self.layer_inputs_index  = []  # list[ LongTensor[E_d] ]
        self.layer_outputs_index = []  # list[ LongTensor[E_d] ]
        self.layer_h    = []           # list[ FloatTensor[E_d,H] ]
        self.layer_c    = []           # list[ FloatTensor[E_d,H] ]
        self.layer_prob = []           # list[ FloatTensor[E_d] ]

        # 输出（与老版 forward 对齐）
        self.weight_list       = []    # list[ FloatTensor[E_d] ]
        self.target_index_list = []    # list[ LongTensor[E_d] ]
        self.prob_list         = []    # list[ FloatTensor[E_d] ]

    # --------- 段工具 ----------
    @staticmethod
    def build_cartesian_pairs(cand_ptr: torch.Tensor, in_ptr: torch.Tensor):
        """把候选段与孩子段做分段笛卡尔积，返回两条对齐索引."""
        device = cand_ptr.device
        K = cand_ptr.numel() - 1
        pcs, phs = [], []
        for i in range(K):
            a0, a1 = cand_ptr[i].item(), cand_ptr[i+1].item()
            b0, b1 = in_ptr[i].item(),  in_ptr[i+1].item()
            Ma, Mb = a1 - a0, b1 - b0
            if Ma == 0 or Mb == 0:
                continue
            pc = torch.arange(a0, a1, device=device).repeat_interleave(Mb)
            ph = torch.arange(b0, b1, device=device).repeat(Ma)
            pcs.append(pc); phs.append(ph)
        if len(pcs) == 0:
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device))
        return torch.cat(pcs, dim=0), torch.cat(phs, dim=0)

    @staticmethod
    def expand_ptr_to_ids(ptr: torch.Tensor):
        """把 CSR ptr 展开为每条元素所属段id."""
        device = ptr.device
        K = ptr.numel() - 1
        if K == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        seglens = (ptr[1:] - ptr[:-1]).to(torch.long)
        return torch.arange(K, device=device).repeat_interleave(seglens)

    # ------------------------- 第 0 层初始化 -------------------------
    def _bootstrap_nodes_layer0(self, device):
        """仅创建第0层的节点 (根节点)；孩子与注意力在扩展时再构造。"""
        predict_h, _, _, predict_date = self.predict_fact
        s0, e0 = int(predict_date[0]), int(predict_date[1])
        d0 = torch.tensor([[s0, e0]], dtype=torch.long, device=device)  # [1,2]
        h0 = torch.tensor([predict_h], dtype=torch.long, device=device) # [1]
        self.nodes_h.append(h0)
        self.nodes_date.append(d0)

    # ------------------------- 单层并行展开 -------------------------
    def _expand_one_layer(self, deep: int, device: str):
        h_d = self.nodes_h[deep]     # [K]
        t_d = self.nodes_date[deep]  # [K,2]
        K = h_d.numel()

        # === A) 构造“上一层孩子CSR” ===
        if deep == 0:
            # 第0层的孩子信息延后到拿到候选总数M后构造（做全对全）
            prev_h = prev_c = prev_p = None
            in_ptr = in_eids = None
        else:
            # 取上一层的边与其弧头（=当前层节点下标）
            prev_h_full   = self.layer_h[deep-1]              # [E_prev,H]
            prev_c_full   = self.layer_c[deep-1]              # [E_prev,H]
            prev_p_full   = self.layer_prob[deep-1]           # [E_prev]
            outputs_idx   = self.layer_outputs_index[deep-1]  # [E_prev]

            # 过滤无效弧头
            if outputs_idx.numel() > 0:
                valid_mask = (outputs_idx >= 0) & (outputs_idx < K)
                if not torch.all(valid_mask):
                    keep = torch.nonzero(valid_mask, as_tuple=False).flatten()
                    outputs_idx = outputs_idx[keep]
                    prev_h_full = prev_h_full[keep]
                    prev_c_full = prev_c_full[keep]
                    prev_p_full = prev_p_full[keep]

            if outputs_idx.numel() == 0:
                # 空孩子集
                in_ptr  = torch.zeros(K+1, dtype=torch.long, device=device)
                in_eids = torch.empty(0, dtype=torch.long, device=device)
                prev_h, prev_c, prev_p = prev_h_full, prev_c_full, prev_p_full
            else:
                # **关键修复：统一排序 + 同步重排 prev_***
                order = torch.argsort(outputs_idx)                      # 排序后的边顺序
                sorted_heads = outputs_idx[order]                       # [E_prev']
                counts = torch.bincount(sorted_heads, minlength=K)      # [K]
                in_ptr = torch.empty(K+1, dtype=torch.long, device=device)
                in_ptr[0] = 0
                in_ptr[1:] = torch.cumsum(counts, dim=0)
                # 孩子边 id 空间与 prev_* 对齐（0..E'-1）
                in_eids = torch.arange(order.numel(), dtype=torch.long, device=device)

                # 同步 prev_* 到排序空间（保持与 in_ptr 一致）
                prev_h = prev_h_full[order]
                prev_c = prev_c_full[order]
                prev_p = prev_p_full[order]

        # === B) 批量候选（来自 KG 接口） ===
        cand = self.kg.enumerate_candidates_batch(
            deep=deep,
            src_h=h_d, src_date=t_d,
            head_relation=self.head_relation,
            predict_fact=self.predict_fact,
            predict_fact_rev=self.predict_fact_rev
        )
        cand_ptr      = cand["cand_ptr"].to(device)        # [K+1] 每个源节点的候选段
        cand_v_h      = cand["cand_v_h"].to(device)        # [M]
        cand_v_date   = cand["cand_v_date"].to(device)     # [M,2]
        cand_allen    = cand["cand_allen"].to(device)      # [M]
        cand_allen_hd = cand["cand_allen_head"].to(device) # [M]
        M = cand_v_h.numel()

        if M == 0:
            # 空层占位
            H = self.attn.hidden_size
            self.layer_inputs_index.append(torch.empty(0, dtype=torch.long, device=device))
            self.layer_outputs_index.append(torch.empty(0, dtype=torch.long, device=device))
            self.layer_h.append(torch.empty(0, H, device=device))
            self.layer_c.append(torch.empty(0, H, device=device))
            self.layer_prob.append(torch.empty(0, device=device))
            self.nodes_h.append(torch.empty(0, dtype=torch.long, device=device))
            self.nodes_date.append(torch.empty(0, 2, dtype=torch.long, device=device))
            self.weight_list.append(torch.empty(0, device=device))
            self.target_index_list.append(torch.empty(0, dtype=torch.long, device=device))
            self.prob_list.append(torch.empty(0, device=device))
            return

        # === C) 第0层：补全“全对全”的孩子集合（关键修复点一） ===
        if deep == 0:
            H = self.attn.hidden_size
            # 伪孩子数量=M，prev_* 全零/全1；in_ptr=[0,M]，in_eids=arange(M)
            prev_h = torch.zeros(M, H, device=device)
            prev_c = torch.zeros(M, H, device=device)
            prev_p = torch.ones(M, device=device)
            in_ptr  = torch.tensor([0, M], dtype=torch.long, device=device)
            in_eids = torch.arange(M, dtype=torch.long, device=device)

        # === D) 段编号 & 候选×孩子配对 ===
        cand_src_node  = self.expand_ptr_to_ids(cand_ptr)   # [M] 每条候选来自哪个源节点
        child_src_node = self.expand_ptr_to_ids(in_ptr)     # [E_prev'] 每条孩子边属于哪个源节点
        pair_cand_idx, pair_child_off = self.build_cartesian_pairs(cand_ptr, in_ptr)  # [L],[L]
        pair_child_idx = in_eids[pair_child_off] if in_eids.numel() > 0 else pair_child_off  # [L]

        # === E) 嵌入 ===
        allen_emb      = self.attn.allen_embedding(cand_allen)         # [M,D1]
        allen_head_emb = self.attn.allen_head_embedding(cand_allen_hd) # [M,D2]
        x_emb = torch.cat([allen_emb, allen_head_emb], dim=-1)         # [M,H]

        # === F) 段注意力、Pruner、TreeLSTM 前向 ===
        attn_w, h_hat = self.attn.rnn.segment_attention(
            x_emb=x_emb,
            child_h=prev_h,
            pair_cand_idx=pair_cand_idx,
            pair_child_idx=pair_child_idx
        )

        x_pruner = torch.cat([x_emb, h_hat], dim=-1)                   # [M,in_dim]
        s, p = self.attn.pruner.segment_forward(
            x_pruner=x_pruner,
            attention_weight=attn_w,
            pre_edge_prob=prev_p,
            pair_cand_idx=pair_cand_idx,
            pair_child_idx=pair_child_idx,
            child_src_node=child_src_node
        )

        now_h, now_c = self.attn.rnn.segment_forward(
            x_emb=x_emb,
            child_h=prev_h,
            child_c=prev_c,
            h_hat=h_hat,
            pair_cand_idx=pair_cand_idx,
            pair_child_idx=pair_child_idx
        )

        # === G) 全层 top-k（与老版一致） ===
        topk_num = min(self.TopK, M)
        _, topk_idx = torch.topk(s, topk_num)
        topk_idx, _ = torch.sort(topk_idx)  # 稳定顺序

        inputs_idx = cand_src_node[topk_idx]     # 本层每条边来自哪个源节点
        v_h_sel    = cand_v_h[topk_idx]          # 本层边的尾实体 id（未去重）
        v_date_sel = cand_v_date[topk_idx]       # 本层边的尾实体时间（未去重）
        layer_h    = now_h[topk_idx]
        layer_c    = now_c[topk_idx]
        layer_prob = p[topk_idx]                 # 传播概率（logit 空间）

        # === H) 按 (v_h, v_s, v_e) 去重得到下一层节点；inverse 作为本层边的 outputs_index（等价老版）
        v_keys = torch.cat([v_h_sel.unsqueeze(1), v_date_sel], dim=1)  # [k,3]
        v_unique, inverse = torch.unique(v_keys, dim=0, return_inverse=True)  # sorted=False 保持首次出现顺序
        outputs_idx = inverse                                           # [k] 本层边的弧头在下一层节点表中的下标

        # 保存下一层节点
        self.nodes_h.append(v_unique[:, 0].contiguous())        # [K_next]
        self.nodes_date.append(v_unique[:, 1:3].contiguous())   # [K_next,2]

        # 保存本层边（索引不丢）
        self.layer_inputs_index.append(inputs_idx.contiguous())
        self.layer_outputs_index.append(outputs_idx.contiguous())
        self.layer_h.append(layer_h.contiguous())
        self.layer_c.append(layer_c.contiguous())
        self.layer_prob.append(layer_prob.contiguous())

        # === I) 计算与头关系贴合的权重（与老版一致） ===
        head_relation = torch.tensor(self.head_relation, dtype=torch.long, device=device)
        h_trans = self.attn.trans_matrix_list[deep](layer_h)                        # [k,H]
        head_emb = self.attn.head_relation_len_embedding_list[deep](head_relation)  # [H]
        weight = (F.cosine_similarity(h_trans, head_emb.unsqueeze(0), dim=-1) + 1.0) / 2.0
        weight = weight * self.attn.head_relation_len_weight[deep]

        self.weight_list.append(weight.contiguous())
        self.target_index_list.append(v_h_sel.contiguous())   # 与老版相同：每条边的尾实体 id
        self.prob_list.append(layer_prob.contiguous())

    # ------------------------- 主流程 -------------------------
    def forward(self, MAX_LEN: int, device: str = "cuda"):
        """
        返回：
          weight_list        : List[Tensor[k_d]]      —— 每条保留边的语义贴合权重
          target_index_list  : List[LongTensor[k_d]] —— 每条保留边的尾实体 id
          prob_list          : List[Tensor[k_d]]      —— 每条保留边的传播概率（logit）
        """
        if len(self.nodes_h) == 0:
            self._bootstrap_nodes_layer0(device)

        for d in range(MAX_LEN):
            self._expand_one_layer(d, device=device)
            if self.layer_h[-1].numel() == 0:
                break

        return self.weight_list, self.target_index_list, self.prob_list
