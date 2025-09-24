import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool, MessagePassing, BatchNorm
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax, coalesce, add_remaining_self_loops
from torch_geometric.nn.norm import LayerNorm

class SimpleAtomEncoder(nn.Module):
    def __init__(self, emb_dim, field_cardinalities):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(C, emb_dim) for C in field_cardinalities])
        for emb in self.embs:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, x_cat): 
        out = 0
        for i in range(x_cat.size(1)):
            out = out + self.embs[i](x_cat[:, i])
        return out 

class SimpleBondEncoder(nn.Module):
    def __init__(self, emb_dim, field_cardinalities):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(C, emb_dim) for C in field_cardinalities])
        for emb in self.embs:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, e_cat): 
        out = 0
        for i in range(e_cat.size(1)):
            out = out + self.embs[i](e_cat[:, i])
        return out 

class GINConvMol(MessagePassing):
    def __init__(self, nn_module, bond_encoder, eps=0.0, train_eps=False):
        super().__init__(aggr='add')
        self.nn = nn_module
        self.bond_encoder = bond_encoder
        self.initial_eps = eps
        self.eps = nn.Parameter(torch.tensor([eps])) if train_eps else eps
        self.train_eps = train_eps

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = ((1 + self.eps) * x) + out
        return self.nn(out)

    def message(self, x_j, edge_attr):
        e = self.bond_encoder(edge_attr)  # [E, hidden]
        return (x_j + e).relu()



class GINConvWithEdgeWeight(MessagePassing):
    def __init__(self, nn_module, eps=0.0, train_eps=False):
        super().__init__(aggr='add')
        self.nn = nn_module
        self.initial_eps = eps
        self.eps = torch.nn.Parameter(torch.Tensor([eps])) if train_eps else eps
        self.train_eps = train_eps

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        device, dtype = x.device, x.dtype
        pre = check_double_self_loops(edge_index, edge_weight, num_nodes=x.size(0))

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=device)

        edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes=num_nodes)

        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
        )

        post = check_double_self_loops(edge_index, edge_weight, num_nodes=x.size(0))

        return self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_weight=edge_weight))

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return f'{self.__class__.__name__}(nn={self.nn})'
    

class GINEncoderWithEdgeWeight(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout=0.5, train_eps=False,
                 global_pooling='mean', dataset_name=None,
                 atom_field_cardinalities=None, bond_field_cardinalities=None,
                 vgin=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.convs, self.bns, self.acts = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.global_pooling = global_pooling
        self.dropout = float(dropout)
        self.vgin = False

        self.use_mol = (dataset_name is not None) and (dataset_name.lower() in {'goodhiv','molhiv','ogbg-molhiv'})
        if self.use_mol:
            assert atom_field_cardinalities is not None and bond_field_cardinalities is not None
            self.atom_encoder = SimpleAtomEncoder(hidden_dim, atom_field_cardinalities)
            self.bond_encoder = SimpleBondEncoder(hidden_dim, bond_field_cardinalities)
        else:
            self.atom_encoder = None
            self.bond_encoder = None

        if self.vgin:
            self.virtual_node_emb = nn.Embedding(1, hidden_dim)
            self.virtual_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim), nn.BatchNorm1d(2 * hidden_dim), nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                nn.Dropout(self.dropout)
            )

        for layer in range(num_layers):
            input_dim = (hidden_dim if self.use_mol else (in_dim if layer == 0 else hidden_dim))
            mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            conv = (GINConvMol(mlp, bond_encoder=self.bond_encoder, train_eps=train_eps)
                    if self.use_mol else
                    GINConvWithEdgeWeight(mlp, train_eps=train_eps))
            self.convs.append(conv); self.bns.append(BatchNorm(hidden_dim)); self.acts.append(nn.ReLU())

    def forward(self, x, edge_index, edge_weight=None, batch=None, node_weight=None, edge_attr=None):
        if self.vgin and batch is None:
            raise ValueError("vGIN requires 'batch' (graph ids per node).")

        if self.use_mol:
            if x.dtype in (torch.int32, torch.int64): x = self.atom_encoder(x)
            elif x.dtype in (torch.float32, torch.float64):
                if x.size(-1) != self.hidden_dim:
                    raise ValueError(f"[Mol path] float x dim={x.size(-1)} != hidden_dim={self.hidden_dim}")
            else:
                raise ValueError(f"[Mol path] Unsupported x dtype: {x.dtype}")
            if node_weight is not None: x = x * node_weight.view(-1,1)

        if self.vgin:
            num_graphs = int(batch.max().item()) + 1
            v = self.virtual_node_emb.weight[0].unsqueeze(0).expand(num_graphs, -1)  # [G, H]

        for li, (conv, bn, act) in enumerate(zip(self.convs, self.bns, self.acts)):
            if self.vgin:
                x = x + v[batch]   # [N,H] += [N,H]

            if self.use_mol:
                x = conv(x, edge_index, edge_attr=edge_attr)            # no self-loops on mol path
            else:
                x = conv(x, edge_index, edge_weight=edge_weight)        # your original path

            x = bn(x); x = act(x); x = F.dropout(x, p=self.dropout, training=self.training)

            if self.vgin and li < len(self.convs) - 1:  # update every layer
                pooled = global_add_pool(x, batch) + v                  # [G,H]
                v = self.virtual_mlp(pooled)                            # [G,H]

        if self.global_pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.global_pooling == 'sum':
            x = global_add_pool(x, batch)
        elif self.global_pooling == 'none':
            pass
        else:
            raise ValueError(f"Unsupported pooling type: {self.global_pooling}")
        return x

    
    
class ExpertClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout, global_pooling):
        super().__init__()
        self.encoder = GINEncoderWithEdgeWeight(
            hidden_dim, hidden_dim, 1, dropout,
            train_eps=True, global_pooling=global_pooling
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        h = self.encoder(x, edge_index, edge_weight, batch)  # → [B, hidden_dim]
        out = self.mlp(h)  # → [B, num_classes]
        return out
    


def check_double_self_loops(edge_index,
                            edge_weight,
                            num_nodes,
                            raise_on_duplicates = True):
    """
    Returns a dict summary and (optionally) raises if any node has >1 self-loop.
    edge_index: (2, E)
    edge_weight: (E,) or None
    num_nodes: optional but recommended (for complete bincount)
    """
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index must be (2, E)"
    E = edge_index.size(1)

    if edge_weight is not None:
        assert edge_weight.dim() == 1 and edge_weight.numel() == E, \
            f"edge_weight must be shape (E,), got {edge_weight.shape} vs E={E}"

    # self-loop mask
    self_mask = edge_index[0].eq(edge_index[1])  # (E,)
    self_nodes = edge_index[0, self_mask]        # nodes with self-loops (with multiplicity)
    n_self_edges = int(self_mask.sum())

    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if E > 0 else 0

    # count how many self-loops each node has
    per_node_counts = torch.bincount(self_nodes, minlength=num_nodes) if n_self_edges > 0 \
                      else torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)

    dup_nodes = (per_node_counts > 1).nonzero(as_tuple=False).view(-1)
    has_duplicates = dup_nodes.numel() > 0
    max_self_loops_on_a_node = int(per_node_counts.max().item()) if n_self_edges > 0 else 0

    summary = {
        "num_edges": E,
        "num_nodes": num_nodes,
        "num_self_loop_edges": n_self_edges,
        "num_nodes_with_self_loop": int((per_node_counts > 0).sum().item()),
        "max_self_loops_on_a_node": max_self_loops_on_a_node,
        "has_duplicate_self_loops": bool(has_duplicates),
        "duplicate_nodes": dup_nodes.detach().cpu().tolist(),  # nodes with >1 self-loop
    }

    if has_duplicates and raise_on_duplicates:
        raise ValueError(f"Duplicate self-loops detected on nodes {summary['duplicate_nodes']}. "
                         f"Max per node = {summary['max_self_loops_on_a_node']}.")

    return summary