import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import BatchNorm as PYG_BatchNorm
from models.gnn_models import GINEncoderWithEdgeWeight, ExpertClassifier
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from contextlib import contextmanager
import math

class Experts(nn.Module):
    """
    Multi-expert module that:
      - learns hard-concrete node/edge/feature masks per expert;
      - computes per-expert logits;
      - applies (a) gate-agnostic regularization and diversity losses,
               (b) LA (label-adversarial) on semantic residual,
               (c) VIB on h_C (causal semantic) instead of EA,
               (d) (scheduled) structural invariance weight (_weight_str_live), even if str_loss is currently 0.0.
    """
    def __init__(self, config, dataset_info):
        super().__init__()

        self.num_features = dataset_info['num_features']
        self.num_classes  = dataset_info['num_classes']
        self.dataset_name = (config.get('dataset').get('dataset_name')).lower()
        self.metric = dataset_info['metric']
        self.is_mol = self.dataset_name in {"goodhiv", "molhiv", "ogbg-molhiv"}
        self.vgin = False
        self.atom_field_cardinalities = None
        self.bond_field_cardinalities = None
        if self.is_mol:
            self.atom_field_cardinalities = [len(v) for v in x_map.values()]
            self.bond_field_cardinalities = [len(v) for v in e_map.values()]
        if self.metric == "Accuracy" and self.num_classes == 1:
            self.num_classes = 2

        mcfg         = config.get('model', {})
        hidden_dim   = mcfg['hidden_dim']
        num_layers   = mcfg['num_layers']
        dropout      = mcfg['dropout']

        mcfg_bn = config.get('model', {})
        self.global_pooling = mcfg_bn.get('global_pooling', 'mean')

        self.num_experts = mcfg['num_experts']
        self.verbose    = config['experiment']['debug']['verbose']

        self.weight_ce  = float(mcfg['weight_ce'])
        self.weight_reg = float(mcfg['weight_reg'])
        self.weight_div = float(mcfg['weight_div'])

        self.causal_encoder = GINEncoderWithEdgeWeight(
            self.num_features, hidden_dim, num_layers, dropout, train_eps=True, global_pooling='none', dataset_name=self.dataset_name, atom_field_cardinalities=self.atom_field_cardinalities, bond_field_cardinalities=self.bond_field_cardinalities, vgin=self.vgin
        )

        self.expert_edge_masks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(self.num_experts)
        ])

        self.classifier_encoders = nn.ModuleList([
            GINEncoderWithEdgeWeight(
                self.num_features, hidden_dim, num_layers, dropout, train_eps=True, global_pooling=self.global_pooling, dataset_name=self.dataset_name, atom_field_cardinalities=self.atom_field_cardinalities, bond_field_cardinalities=self.bond_field_cardinalities, vgin=self.vgin)
        for _ in range(self.num_experts)
        ])

        self.expert_classifiers_causal = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_classes)
            )
            for _ in range(self.num_experts)
        ])

        self.rho_edge_config = config.get('model', {}).get('rho_edge', 0.5)
        self.rho_edge = nn.Parameter(torch.full((self.num_experts,), self.rho_edge_config))

    def set_epoch(self, epoch: int):
        """Update schedulers."""
        t = max(epoch, 0)
        if self.verbose and epoch % 10 == 0:
            print(f"[Experts.set_epoch] epoch={epoch}")

    def forward(self, data, target=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        Z = self.causal_encoder(x, edge_index, batch=batch, edge_attr=getattr(data, "edge_attr", None))
        edge_feat = torch.cat([Z[edge_index[0]], Z[edge_index[1]]], dim=1)

        node_masks, edge_masks = [], []
        expert_logits, h_stable_list = [], []

        is_eval = not self.training

        for k in range(self.num_experts):
            edge_mask_logits = self.expert_edge_masks[k](edge_feat)
            
            edge_mask = self._hard_concrete_mask(edge_mask_logits, is_eval=is_eval)

            src, dst = edge_index                
            e_on = edge_mask.view(-1).to(      
                dtype=torch.float32, device=x.device
            )

            N = x.size(0)
            node_weight = e_on.new_zeros(N)      
            node_weight.index_add_(0, src, e_on)
            node_weight.index_add_(0, dst, e_on)
            node_weight = (node_weight > 0).float()
            node_masks.append(node_weight.view(-1, 1))
            edge_masks.append(e_on.view(-1, 1))                    
            masked_x   = x * node_weight.view(-1, 1)    
            edge_weight = e_on          

            if self.is_mol:
                h_stable = self.classifier_encoders[k](x, edge_index, edge_weight=edge_weight, batch=batch, node_weight=node_weight, edge_attr=getattr(data, "edge_attr", None))
            else:
                h_stable = self.classifier_encoders[k](masked_x, edge_index, edge_weight=edge_weight, batch=batch, edge_attr=getattr(data, "edge_attr", None))
            logit = self.expert_classifiers_causal[k](h_stable)

            h_stable_list.append(h_stable)
            expert_logits.append(logit)

        node_masks    = torch.stack(node_masks, dim=1)        # (N, K, 1)
        edge_masks    = torch.stack(edge_masks, dim=1)        # (E, K, 1)
        expert_logits = torch.stack(expert_logits, dim=1)     # (B, K, C)
        h_stable_list = torch.stack(h_stable_list, dim=1)     # (B, K, H)

        if self.global_pooling == 'mean':
            h_orig = global_mean_pool(Z, batch)            # (B, H)
        elif self.global_pooling == 'sum':
            h_orig = global_add_pool(Z, batch)            # (B, H)  

        out = {
            'h_stable': h_stable_list,
            'h_orig':   h_orig,
            'node_masks': node_masks,
            'edge_masks': edge_masks,
            'expert_logits': expert_logits,
            'rho': [self.rho_node, self.rho_edge]
        }

        if target is not None:
            ce_list, reg_list, tot_list = [], [], []

            B = expert_logits.size(0)
            K = self.num_experts
            ce_ps  = []

            for k in range(self.num_experts):
                logits_k = expert_logits[:, k, :]
                hC_k     = h_stable_list[:, k, :]
                node_mask_k = node_masks[:, k, :]
                edge_mask_k = edge_masks[:, k, :]

                pred = logits_k.squeeze(-1).float()
                y = target.float()
                ce_vec = F.l1_loss(pred, y, reduction='none') * self.weight_ce
                ce = ce_vec.mean()
                ce_ps.append(ce_vec)
                ce_list.append(ce)

                reg = self._mask_reg(
                    node_masks[:, k, :], edge_masks[:, k, :],
                    node_batch=batch, edge_batch=batch[edge_index[0]], expert_idx=k
                ) * self.weight_reg
                reg_list.append(reg)

                total = (ce + reg)
                tot_list.append(total)

            ce_ps  = torch.stack(ce_ps,  dim=1)   # (B,K)
            div_loss = self.weight_div * self._diversity_loss(node_masks, edge_masks,
                                            node_batch=batch, edge_batch=batch[edge_index[0]])

            out.update({
                'loss_total_list': torch.stack(tot_list),   # (K,)
                'loss_ce_list':    torch.stack(ce_list),    # (K,)
                'loss_reg_list':   torch.stack(reg_list),   # (K,)
                'loss_div':        div_loss,
                'per_sample': {
                    'ce':  ce_ps,
                }
            })

        return out

    # ----------------- Internals -----------------
    def _hard_concrete_mask(self, logits, temperature=0.1, is_eval=False):
        if self.training:
            u = torch.rand_like(logits)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            y_soft = torch.sigmoid((logits + g) / 0.1)
        else:
            u = torch.rand_like(logits)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            y_soft = torch.sigmoid((logits + g) / 0.1)

        y_hard = (y_soft > 0.5).float()
        return y_hard + (y_soft - y_soft.detach())

    def _ce(self, pred, target, use_weights=False):
        if use_weights:
            C = pred.size(1)
            counts = torch.bincount(target, minlength=C).float()
            counts[counts == 0] = 1.0
            w = (1.0 / counts)
            w = w / w.sum()
            return F.cross_entropy(pred, target, weight=w.to(pred.device))
        return F.cross_entropy(pred, target)
    
    def _reg(self, pred, target):
        return F.l1_loss(pred, target)
    
    def _diversity_loss(
        self,
        node_masks: torch.Tensor,   # [N_nodes, K, 1] or [N_nodes, K]
        edge_masks: torch.Tensor,   # [N_edges, K, 1] or [N_edges, K]
        node_batch: torch.Tensor,   # [N_nodes] graph indices (0..B-1)
        edge_batch: torch.Tensor,   # [N_edges] graph indices (0..B-1)
    ):
        """
        Combines three complementary pieces:
        1) correlation hinge (shape-level de-correlation across experts)
        2) union/coverage (avoid "nobody selects anything")
        3) overlap/exclusivity (avoid "everybody selects the same thing")

        Hyperparameters are kept local with sensible defaults.
        """

        w_corr = 1.0
        tau_corr = 0.10

        eps = 1e-8

        def _maybe_squeeze(v: torch.Tensor) -> torch.Tensor:
            return v.squeeze(-1) if v.dim() >= 2 and v.size(-1) == 1 else v

        def _per_graph_abs_corr_hinge(V: torch.Tensor, bidx: torch.Tensor) -> torch.Tensor:
            """
            V: [items, K] mask probabilities for one modality
            For each graph, z-score over items, compute |corr| matrix across experts,
            take off-diagonal mean, hinge above tau_corr, then average over graphs.
            """
            if V.numel() == 0 or bidx.numel() == 0:
                return V.new_tensor(0.0)

            B = int(bidx.max().item()) + 1 if bidx.numel() > 0 else 0
            vals = []
            for g in range(B):
                sel = (bidx == g)
                if sel.sum() < 2:
                    continue
                X = V[sel]  # [n_g, K]
                # standardize over items to avoid trivial scale effects
                X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, unbiased=False, keepdim=True) + eps)
                # correlation proxy via normalized inner product
                C = (X.t() @ X) / max(X.size(0), 1)  # [K,K]
                M = C.abs()
                # off-diagonal mean
                off_sum = M.sum() - M.diag().sum()
                denom   = max(M.size(0) * (M.size(0) - 1), 1)
                off_mean = off_sum / denom
                vals.append(F.relu(off_mean - tau_corr))
            return torch.stack(vals).mean() if len(vals) else V.new_tensor(0.0)

        N = _maybe_squeeze(node_masks)                   # [N_nodes, K]
        E = _maybe_squeeze(edge_masks)                   # [N_edges, K]

        # -------- compute components per modality --------
        n_corr = _per_graph_abs_corr_hinge(N, node_batch)
        e_corr = _per_graph_abs_corr_hinge(E, edge_batch)
        corr   = torch.stack([n_corr, e_corr]).mean()

        loss = w_corr * corr
        return loss

    def _mask_reg(self, node_mask, edge_mask, node_batch, edge_batch, expert_idx: int,
                  use_fixed_rho: bool = False, fixed_rho_vals: tuple = (0.5, 0.5)):
        if use_fixed_rho:
            rho_node, rho_edge = [float(min(max(v, 0.0), 1.0)) for v in fixed_rho_vals]
        else:
            rho_edge = self.rho_edge_config

        def per_graph_keep(mask_vals, batch_idx):
            G = batch_idx.max().item() + 1
            keep = torch.zeros(G, device=mask_vals.device)
            cnt  = torch.zeros(G, device=mask_vals.device)
            keep.scatter_add_(0, batch_idx, mask_vals.squeeze())
            cnt.scatter_add_(0, batch_idx, torch.ones_like(mask_vals.squeeze()))
            return keep / (cnt + 1e-8)

        node_keep_pg = per_graph_keep(node_mask, node_batch)
        edge_keep_pg = per_graph_keep(edge_mask, edge_batch)

        return ((edge_keep_pg - rho_edge) ** 2).mean() * 5.0

    @contextmanager
    def _frozen_params(self, module: nn.Module, freeze_bn_running_stats: bool = True):
        was_training = module.training
        if freeze_bn_running_stats:
            module.eval()
        saved = [p.requires_grad for p in module.parameters()]
        for p in module.parameters():
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p, rg in zip(module.parameters(), saved):
                p.requires_grad_(rg)
            module.train(was_training)

x_map = {
    'atomic_num':
        list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
        list(range(0, 11)),
    'formal_charge':
        list(range(-5, 7)),
    'num_hs':
        list(range(0, 9)),
    'num_radical_electrons':
        list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
    }

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}