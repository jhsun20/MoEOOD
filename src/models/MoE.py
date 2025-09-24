import torch
import torch.nn as nn
import torch.nn.functional as F

from entmax import entmax_bisect
from torch_geometric.nn import global_mean_pool

from models.expert import Experts
from models.gnn_models import GINEncoderWithEdgeWeight


class MoE(nn.Module):
    def __init__(self, config, dataset_info):
        super().__init__()
        self.verbose         = config['experiment']['debug']['verbose']
        self.num_experts     = config['model']['num_experts']
        self.aggregation     = config['model']['aggregation']
        self.train_after     = config['gate']['train_after']
        self.weight_ce       = config['model']['weight_ce']
        self.weight_reg      = config['model']['weight_reg']
        self.weight_la      = config['model']['weight_la']
        self.weight_ea      = config['model']['weight_ea']
        self.weight_str      = config['model']['weight_str']
        self.weight_div      = config['model']['weight_div']
        self.weight_load     = config['model']['weight_load']

        self.shared = Experts(config, dataset_info)

        self.num_features   = dataset_info['num_features']
        self.dataset_name   = dataset_info['dataset_name']
        self.num_classes    = dataset_info['num_classes']
        if self.num_classes == 1 and dataset_info['metric'] == "Accuracy":
            self.num_classes = 2
        self.metric         = dataset_info['metric']

        gate_hidden    = config['gate']['hidden_dim']
        gate_depth     = config['gate']['depth']
        dropout        = config['model']['dropout']
        self._gate_mlp = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.entmax_alpha = float(config['gate']['entmax_alpha'])
        gcfg = config.get('gate', {})
        self.gate_tau_oracle = float(gcfg.get('tau_oracle', 0.75))
        self.gate_temperature = float(gcfg.get('temperature', 1.0))
        self.weight_gate      = float(gcfg.get('weight_gate', 0.1))
        self.weight_gate_oracle = float(gcfg.get('weight_oracle', 1.0))
        self.gate_teacher_w_ea = float(gcfg.get('teacher_w_ea', 1.0))
        self.gate_teacher_w_la = float(gcfg.get('teacher_w_la', 1.0))
        self.current_epoch = 0

        if self.verbose:
            print(f"[MoE] K={self.num_experts}, aggregation={self.aggregation}, gate=raw-graph")

    # ------------- Public API -------------
    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)
        self.shared.set_epoch(epoch)

    def forward(self, data):
        shared_out = self.shared(data, data.y)

        gate_scores = self._gate_logits_expert_features(data, shared_out)
        gate_loss, p_gate_train = self._compute_gate_loss(shared_out, data.y, gate_scores)

        if self.current_epoch < self.train_after:
            gate_probs = torch.full_like(gate_scores, 1.0 / self.num_experts)
        else:
            gate_probs = p_gate_train

        if self.verbose:
            with torch.no_grad():
                print("[MoE] gate (first row):", gate_probs[0].tolist())

        out = self._aggregate(shared_out, gate_probs, targets=data.y)

        out['loss_gate']  = self.weight_gate * gate_loss
        out['loss_total'] = out['loss_total'] + out['loss_gate']
        out['p_gate_train'] = p_gate_train

        return out

    # ------------- Internals -------------
    def _gate_logits_raw(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.gate_enc(x, edge_index, batch=batch)
        return self.gate_mlp(h)

    def _aggregate(self, shared_out, gate_probs, targets):
        expert_logits = shared_out['expert_logits']
        B, K, C = expert_logits.shape

        weighted = expert_logits * gate_probs.unsqueeze(-1)
        agg_logits = weighted.sum(dim=1)

        ce = self._gate_weighted_ce(expert_logits.transpose(0, 1), targets, gate_probs.transpose(0, 1))

        ps = shared_out.get('per_sample', None)
        def gate_weighted_mean(mat_bk):
            return (gate_probs * mat_bk).sum(dim=1).mean()

        avg_gate = gate_probs.mean(dim=0).detach()
        reg = (avg_gate * shared_out['loss_reg_list']).sum()

        div = shared_out['loss_div']

        total = (ce + reg + div)

        return {
            'logits': agg_logits,
            'loss_total': total,    
            'loss_ce': ce,
            'loss_reg': reg,
            'loss_div': div,
            'gate_weights': gate_probs,
            'rho': shared_out['rho'],
            'expert_logits': expert_logits,
            'node_masks': shared_out['node_masks'],
            'edge_masks': shared_out['edge_masks']
        }

    def _compute_gate_loss(self, shared_out, targets, gate_scores, eps: float = 1e-12):
        expert_logits = shared_out['expert_logits']
        B, K, C = expert_logits.shape

        with torch.no_grad():

            uni = torch.full((K, B), 1.0 / K, device=expert_logits.device)

            ce_per = self._gate_weighted_ce(
                expert_logits.transpose(0,1), targets, uni, return_matrix=True
            )

            r = -ce_per
            q = F.softmax(r / self.gate_tau_oracle, dim=1)

        scores = gate_scores / max(self.gate_temperature, 1e-6)
        if self.entmax_alpha > 1:
            p = entmax_bisect(scores, alpha=self.entmax_alpha, dim=1)
        else:
            p = F.softmax(scores, dim=1)

        kl = F.kl_div((q.clamp_min(eps)).log(), p.clamp_min(eps), reduction='batchmean')

        lb = self._load_balance(p)

        gate_loss = self.weight_gate_oracle * kl + self.weight_load * lb
        return gate_loss, p
    
    def _gate_logits_expert_features(self, data, shared_out, eps: float = 1e-12):
        logits = shared_out['expert_logits'].detach()
        probs  = logits.softmax(dim=-1)
        B, K, C = probs.shape
        dev, dtype = probs.device, probs.dtype

        maxp    = probs.max(dim=-1).values.unsqueeze(-1)
        top2    = probs.topk(k=min(2, C), dim=-1).values
        margin  = (top2[..., 0] - (top2[..., 1] if C > 1 else 0.)).unsqueeze(-1)
        entropy = (-(probs.clamp_min(eps) * probs.clamp_min(eps).log())
                .sum(-1, keepdim=True))
        energy  = (-torch.logsumexp(logits, dim=-1, keepdim=True))

        weak_kl = shared_out.get('weak_kl_list', None)
        stab    = (-weak_kl.detach()).unsqueeze(-1) if weak_kl is not None else torch.zeros(B,K,1, device=dev, dtype=dtype)

        with torch.no_grad():
            if K > 1:
                logp   = probs.clamp_min(eps).log()
                kl_kj  = (probs.unsqueeze(2) * (logp.unsqueeze(2) - logp.unsqueeze(1))).sum(-1)
                sym_kl = 0.5*(kl_kj + kl_kj.transpose(1,2))
                disagree = -(sym_kl.sum(dim=2) / (K - 1)).unsqueeze(-1)
            else:
                disagree = torch.zeros(B,K,1, device=dev, dtype=dtype)

        pe_c = shared_out.get('env_post_causal', None)
        env_ent = (-(pe_c.clamp_min(eps) * pe_c.clamp_min(eps).log()).sum(-1, keepdim=True)
                if pe_c is not None else torch.zeros(B,K,1, device=dev, dtype=dtype))
        py_s = shared_out.get('label_post_spur', None)
        leak_ent = (-(py_s.clamp_min(eps) * py_s.clamp_min(eps).log()).sum(-1, keepdim=True)
                    if py_s is not None else torch.zeros(B,K,1, device=dev, dtype=dtype))

        n = torch.tensor([data.num_nodes], device=dev, dtype=dtype).view(1,1,1).expand(B,K,1)
        m = torch.tensor([data.num_edges], device=dev, dtype=dtype).view(1,1,1).expand(B,K,1)

        Phi = torch.cat([
            maxp, margin, entropy, energy,
            stab, disagree,
            env_ent, leak_ent,
            n, m
        ], dim=-1)

        mu = Phi.mean(dim=(0,1), keepdim=True)
        sd = Phi.std(dim=(0,1), keepdim=True).clamp_min(1e-3)
        Phi = (Phi - mu) / sd

        d_phi = Phi.size(-1)
        if not hasattr(self, "_gate_mlp") or self._gate_mlp is None:
            self._gate_mlp = nn.Sequential(
                nn.Linear(d_phi, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ).to(dev)

        scores = self._gate_mlp(Phi).squeeze(-1)
        return scores
    
    def _gate_weighted_ce(self, stacked_logits, targets, gate_weights, return_matrix: bool = False):
        K, B, C = stacked_logits.shape
        device = stacked_logits.device
        y = targets.view(-1).long().to(device)
        ce_bk = stacked_logits[0, :, 0].new_zeros(B, K)

        if self.dataset_name != "GOODHIV":
            for k in range(K):
                if self.metric == "Accuracy" or self.metric == "ROC-AUC":
                    ce_bk[:, k] = F.cross_entropy(stacked_logits[k], y, reduction='none')
                else:
                    pred = stacked_logits[k].squeeze(-1).float()
                    target = y.float()
                    ce_bk[:, k] = F.l1_loss(pred, target, reduction='none')
            return ce_bk if return_matrix else (gate_weights * ce_bk.T).sum(dim=0).mean()

        tau_logitadj= 1.0

        counts = torch.bincount(y, minlength=C).float().to(device)
        counts[counts == 0] = 1.0
        prior = (counts / counts.sum()).clamp_min(1e-8)

        for k in range(K):
            logits = stacked_logits[k]
            logits_la = logits + tau_logitadj * prior.log().view(1, -1)
            logp_la    = F.log_softmax(logits_la, dim=1)
            ce_la = F.cross_entropy(logits_la, y, reduction='none') * 10
            ce_all = ce_la
            ce_bk[:, k] = ce_all

        return ce_bk if return_matrix else (gate_weights * ce_bk.T).sum(dim=0).mean()


    @staticmethod
    def _load_balance(gate_probs, lam=0.2, T=0.2, eps=1e-12):
        B, K = gate_probs.shape
        uniform = gate_probs.new_full((K,), 1.0 / K)

        p_bar = gate_probs.mean(dim=0).clamp_min(eps)
        log_pbar = p_bar.log()
        log_u = uniform.log()
        L_bal = torch.sum(uniform * (log_u - log_pbar))   # scalar

        H_rows = -(gate_probs.clamp_min(eps) * gate_probs.clamp_min(eps).log()).sum(dim=1).mean()

        if T <= 0:
            q = torch.zeros_like(gate_probs)
            q.scatter_(1, gate_probs.argmax(dim=1, keepdim=True), 1.0)
        else:
            q_pow = gate_probs.clamp_min(eps).pow(1.0 / T)
            q = q_pow / q_pow.sum(dim=1, keepdim=True)
        counts = q.mean(dim=0).clamp_min(eps)
        L_top1 = torch.sum(uniform * (log_u - counts.log()))

        return L_bal + lam * H_rows + 0.2 * L_top1
