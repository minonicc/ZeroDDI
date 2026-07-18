# -*- coding: utf-8 -*-
import warnings
from tools.registry import Registry
from ...builder import build_left, LEFT
from torch import nn
import torch
import numpy as np
import time
import pandas as pd
import os.path as osp
import pickle as pkl
import os
from types import SimpleNamespace
from rdkit.Chem import AllChem
from .graph_init import mol_to_graph_data_obj_simple
from torch_geometric.nn import global_mean_pool, MessagePassing
#from torch_geometric.nn import GATConv
# from torch_geometric.nn import global_mean_pool,MessagePassing,GCNConv,GINConv,SAGEConv,GATConv
from torch_geometric.data import Batch
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_scatter import scatter_add
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import to_dense_batch
num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3
from torch_geometric.nn import global_add_pool
from .pharmacophore import PharmacophoreExtractor, PharmacophorePairEncoder

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        norm = self.norm(edge_index[0], x.size(0), x.dtype)
        x = self.linear(x)
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__(node_dim=0)
        self.aggr = aggr
        self.heads = heads
        self.emb_dim = emb_dim
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))
        self.bias = nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, heads * emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)

        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
                          self.edge_embedding2(edge_attr[:, 1])

        x = self.weight_linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out += self.bias
        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.gnn_type = gnn_type

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, aggr="add"))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim,aggr="add"))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim, aggr="mean"))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 2:
            x, edge_index = argv[0], argv[1]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation

class SubExtractor(nn.Module):
    def __init__(self, hidden_dim, num_clusters, residual=False):
        super().__init__()
        
        self.Q = nn.Parameter(torch.Tensor(1, num_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.Q)
        
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)

        self.residual = residual
    
    def forward(self, x, batch):
        K = self.W_K(x)
        V = self.W_V(x)
        K, mask = to_dense_batch(K, batch)
        # mask: (batch_size, max_num_nodes)
        V, _ = to_dense_batch(V, batch)
        
        attn_mask = (~mask).float().unsqueeze(1)
        attn_mask = attn_mask * (-1e9)
        
        Q = self.Q.tile(K.size(0), 1, 1)
        Q = self.W_Q(Q)
        
        A = Q @ K.transpose(-1, -2) / (Q.size(-1) ** 0.5)
        A = A + attn_mask
        A = A.softmax(dim=-1)
        # (batch_size, num_clusters, max_num_nodes)
        
        out = Q + A @ V
        
        if self.residual:
            out = out + self.W_O(out).relu()
        else:
            out = self.W_O(out).relu()
        
        return out, A, mask

@LEFT.register_module()
class GNN_model(nn.Module):
    def __init__(
        self,
        Allfilename,
        dropout,
        device,
        num_layer,
        JK,
        gnn_type,
        output_dim,
        extra_data=None,
        sub_number=30,
        use_sub=True,
        use_pharmacophore_pairs=False,
        pharmacophore_pair_dim=300,
        pharmacophore_pair_hidden_dim=300,
        pharmacophore_type_dim=32,
        pharmacophore_max_pairs=128,
        pharmacophore_pooling="sum",
        deduplicate_drugs_in_batch=False,
        cache_drug_graphs_on_device=False,
        batch_pharmacophore_pair_mlp=False,
    ):
        super().__init__()
        """
        gnn_type are: "gin","gcn","gat","graphsage"
        """
        self.Allfilename = Allfilename
        self.mol_dim = 256
        self.device = device
        self.extra_data = extra_data
        self.use_sub = use_sub
        self.use_pharmacophore_pairs = use_pharmacophore_pairs
        self.deduplicate_drugs_in_batch = deduplicate_drugs_in_batch
        self.cache_drug_graphs_on_device = cache_drug_graphs_on_device

        self.num_layer = num_layer
        self.dropout_ratio = dropout
        self.JK = JK
        self.gnn_type = gnn_type
        self.sub_number=sub_number

        self.pharmacophore_extractor = None
        self.name2pharmacophores = {}
        if self.use_pharmacophore_pairs:
            self.pharmacophore_extractor = PharmacophoreExtractor()

        self.name2data = self.get_graph_data()
        self.gnn = GNN(self.num_layer, 300, JK=self.JK, drop_ratio=self.dropout_ratio, gnn_type=self.gnn_type)
        #self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))
        self.linear = nn.Linear(600, self.mol_dim)
        self.Dropout_layer = nn.Dropout(self.dropout_ratio)
        if self.use_sub:
            self.pool = SubExtractor(300, self.sub_number, False)
        if self.use_pharmacophore_pairs:
            self.pharmacophore_pair_encoder = PharmacophorePairEncoder(
                atom_dim=300,
                output_dim=pharmacophore_pair_dim,
                type_dim=pharmacophore_type_dim,
                hidden_dim=pharmacophore_pair_hidden_dim,
                max_pairs=pharmacophore_max_pairs,
                pooling=pharmacophore_pooling,
                dropout=self.dropout_ratio,
                batch_pair_mlp=batch_pharmacophore_pair_mlp,
            )

        if self.cache_drug_graphs_on_device:
            self.name2data = {
                drug: data.to(self.device)
                for drug, data in self.name2data.items()
            }
        

    def get_graph_data(self):    
        dataset = pd.read_csv(self.Allfilename)
        drug1_list = [[a, b] for a, b in zip(dataset["drug1"], dataset["smiles1"])]
        drug2_list = [[a, b] for a, b in zip(dataset["drug2"], dataset["smiles2"])]
        drug_list = drug1_list + drug2_list
        
        if self.extra_data is not None:
            dataset1 = pd.read_csv(self.extra_data)
            drug1_list1 = [[a, b] for a, b in zip(dataset1["drug1"], dataset1["smiles1"])]
            drug2_list1 = [[a, b] for a, b in zip(dataset1["drug2"], dataset1["smiles2"])]
            drug_list1 = drug1_list1 + drug2_list1
            drug_list = drug_list + drug_list1
        self.drugname_smiles = {}
        for d in drug_list:
            self.drugname_smiles[d[0]] = d[1]
        data_dict = {}
        for name, smiles in self.drugname_smiles.items():
            # each example contains a single species
            try:
                rdkit_mol = AllChem.MolFromSmiles(smiles)
                if rdkit_mol != None:  # ignore invalid mol objects
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    id = int(name.split('DB')[1].lstrip('0'))
                    # id = int(zinc_id_list[i].split('DB')[1].lstrip('0'))
                    data.id = torch.tensor(
                        [id])  # id here is zinc id value, stripped of
                    # leading zeros
                    data_dict[name] = data
                    if self.use_pharmacophore_pairs:
                        features = self.pharmacophore_extractor(rdkit_mol)
                        self.name2pharmacophores[name] = (
                            self.pharmacophore_extractor.to_index_cache(features)
                        )

            except:
                continue
        return data_dict

    def _get_drug_data(self, drug):
        data = self.name2data[drug]
        if self.cache_drug_graphs_on_device:
            return data
        return data.to(self.device)
    
 

    def forward(self, inputs):
        # input : drug1,drug2,event_type
        t1 = time.time()
        drug1s = inputs[0]
        drug2s = inputs[1]
        if self.deduplicate_drugs_in_batch:
            unique_drugs = list(dict.fromkeys(list(drug1s) + list(drug2s)))
            unique_drug_to_index = {drug: index for index, drug in enumerate(unique_drugs)}
            unique_drug_list = [self._get_drug_data(drug) for drug in unique_drugs]
            unique_drug_batch = Batch.from_data_list(unique_drug_list)

            unique_x = self.gnn(
                unique_drug_batch.x,
                unique_drug_batch.edge_index,
                unique_drug_batch.edge_attr,
            )
            unique_drug_batch.node_representation = unique_x
            unique_out = global_add_pool(unique_x, unique_drug_batch.batch)
            if self.use_sub:
                unique_pool, unique_A, _ = self.pool(unique_x, unique_drug_batch.batch)
                unique_pool = F.normalize(unique_pool, dim=-1)
            unique_out = self.projection_head(unique_out)

            drug1_indices = torch.tensor(
                [unique_drug_to_index[drug] for drug in drug1s],
                dtype=torch.long,
                device=unique_out.device,
            )
            drug2_indices = torch.tensor(
                [unique_drug_to_index[drug] for drug in drug2s],
                dtype=torch.long,
                device=unique_out.device,
            )
            out1 = unique_out.index_select(0, drug1_indices)
            out2 = unique_out.index_select(0, drug2_indices)
            if self.use_sub:
                pool1 = unique_pool.index_select(0, drug1_indices)
                pool2 = unique_pool.index_select(0, drug2_indices)
                A1 = unique_A.index_select(0, drug1_indices)
                A2 = unique_A.index_select(0, drug2_indices)
        else:
            drug1_list = []
            drug2_list = []
            for i in range(len(drug1s)):
                drug1 = drug1s[i]
                drug2 = drug2s[i]
                drug1_list.append(self._get_drug_data(drug1))
                drug2_list.append(self._get_drug_data(drug2))
            drug1_batch = Batch.from_data_list(drug1_list)
            drug2_batch = Batch.from_data_list(drug2_list)

            x1 = self.gnn(drug1_batch.x, drug1_batch.edge_index, drug1_batch.edge_attr)
            drug1_batch.node_representation = x1
            out1 = global_add_pool(x1, drug1_batch.batch)
            if self.use_sub:
                pool1, A1, _ = self.pool(x1, drug1_batch.batch)
                pool1 = F.normalize(pool1, dim=-1)
            out1 = self.projection_head(out1)

            x2 = self.gnn(drug2_batch.x, drug2_batch.edge_index, drug2_batch.edge_attr)
            drug2_batch.node_representation = x2
            out2 = global_add_pool(x2, drug2_batch.batch)
            if self.use_sub:
                pool2, A2, _ = self.pool(x2, drug2_batch.batch)
                pool2 = F.normalize(pool2, dim=-1)
            out2 = self.projection_head(out2)

        drugpair_feature = torch.cat((out1, out2), 1)
        out2 = self.linear(drugpair_feature)
        out2 = self.Dropout_layer(out2)
        #print("pool2",pool2.shape)
        
        if self.use_pharmacophore_pairs:
            drug1_features = [self.name2pharmacophores.get(drug, []) for drug in drug1s]
            drug2_features = [self.name2pharmacophores.get(drug, []) for drug in drug2s]
            if self.deduplicate_drugs_in_batch:
                drug1_batch = self._pharmacophore_batch_from_unique(
                    unique_drug_batch,
                    drug1_indices,
                )
                drug2_batch = self._pharmacophore_batch_from_unique(
                    unique_drug_batch,
                    drug2_indices,
                )
            pharmacophore_pairs, pharmacophore_mask = self.pharmacophore_pair_encoder(
                drug1_batch,
                drug2_batch,
                drug1_features,
                drug2_features,
            )
            pharmacophore_evidence = {
                "tokens": pharmacophore_pairs,
                "mask": pharmacophore_mask,
            }
        else:
            pharmacophore_evidence = None

        if self.use_sub:
            sub_structure = torch.cat((pool1,pool2),1)
            if not self.use_pharmacophore_pairs:
                return out2, sub_structure, A1, A2
            aux_attention = {
                "drug2_attention": A2,
                "pharmacophore": pharmacophore_evidence,
            }
            return out2, sub_structure, A1, aux_attention
        else:
            if not self.use_pharmacophore_pairs:
                return out2, None, None, None
            aux_attention = {
                "drug2_attention": None,
                "pharmacophore": pharmacophore_evidence,
            }
            return out2, None, None, aux_attention


    def _pharmacophore_batch_from_unique(self, unique_drug_batch, drug_indices):
        ptr = [0]
        node_chunks = []
        unique_ptr = unique_drug_batch.ptr.detach().cpu().tolist()
        for drug_index in drug_indices.detach().cpu().tolist():
            start = unique_ptr[drug_index]
            end = unique_ptr[drug_index + 1]
            node_chunk = unique_drug_batch.node_representation[start:end]
            node_chunks.append(node_chunk)
            ptr.append(ptr[-1] + node_chunk.size(0))

        if node_chunks:
            node_representation = torch.cat(node_chunks, dim=0)
        else:
            node_representation = unique_drug_batch.node_representation.new_zeros(
                (0, unique_drug_batch.node_representation.size(-1))
            )

        return SimpleNamespace(
            ptr=torch.tensor(
                ptr,
                dtype=torch.long,
                device=unique_drug_batch.node_representation.device,
            ),
            node_representation=node_representation,
        )


    


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)
