import os

import torch
from torch import nn


PHARMACOPHORE_FAMILIES = (
    "Hydrophobe",
    "Aromatic",
    "Donor",
    "Acceptor",
    "PosIonizable",
    "NegIonizable",
)

PC_DDI_FAMILY_NAMES = {
    "Hydrophobe": "Hydrophobic center",
    "Aromatic": "Aromatic ring",
    "Donor": "Hydrogen bond donor",
    "Acceptor": "Hydrogen bond acceptor",
    "PosIonizable": "Positive charged group",
    "NegIonizable": "Negative charged group",
}


class PharmacophoreExtractor:
    """Extract PC-DDI-style pharmacophore features from an RDKit molecule."""

    def __init__(self, families=PHARMACOPHORE_FAMILIES, include_centroid=True):
        try:
            from rdkit import RDConfig
            from rdkit.Chem import AllChem
            from rdkit.Chem import ChemicalFeatures
        except ImportError as exc:
            raise ImportError(
                "RDKit is required for pharmacophore extraction."
            ) from exc

        feature_def = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        self.factory = ChemicalFeatures.BuildFeatureFactory(feature_def)
        self.all_chem = AllChem
        self.families = tuple(families)
        self.family_to_idx = {family: idx for idx, family in enumerate(self.families)}
        self.include_centroid = include_centroid

    def _centroid(self, mol, atom_ids):
        if not self.include_centroid:
            return None
        if mol.GetNumConformers() == 0:
            self.all_chem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        coords = []
        for atom_id in atom_ids:
            position = conf.GetAtomPosition(atom_id)
            coords.append((float(position.x), float(position.y), float(position.z)))
        count = float(len(coords))
        return tuple(sum(coord[axis] for coord in coords) / count for axis in range(3))

    def __call__(self, mol):
        features = []
        seen = set()
        for feature in self.factory.GetFeaturesForMol(mol):
            family = feature.GetFamily()
            if family not in self.family_to_idx:
                continue
            atom_ids = tuple(sorted(int(idx) for idx in feature.GetAtomIds()))
            if not atom_ids:
                continue
            key = (family, atom_ids)
            if key in seen:
                continue
            seen.add(key)
            features.append(
                {
                    "family": family,
                    "family_id": self.family_to_idx[family],
                    "pc_ddi_name": PC_DDI_FAMILY_NAMES[family],
                    "atom_ids": atom_ids,
                    "centroid": self._centroid(mol, atom_ids),
                }
            )
        return features

    def to_index_cache(self, features):
        atom_index = []
        atom_owner = []
        atom_count = []
        type_ids = []
        for feature_index, feature in enumerate(features):
            atom_ids = list(feature["atom_ids"])
            if not atom_ids:
                continue
            atom_index.extend(atom_ids)
            atom_owner.extend([feature_index] * len(atom_ids))
            atom_count.append(len(atom_ids))
            type_ids.append(feature["family_id"])

        return {
            "atom_index": torch.tensor(atom_index, dtype=torch.long),
            "atom_owner": torch.tensor(atom_owner, dtype=torch.long),
            "atom_count": torch.tensor(atom_count, dtype=torch.float32),
            "type_ids": torch.tensor(type_ids, dtype=torch.long),
            "num_pharmacophores": len(type_ids),
            "_device_cache": {},
        }


class PharmacophorePairEncoder(nn.Module):
    """Encode cross-drug pharmacophore pairs as DDIE-selectable evidence."""

    def __init__(
        self,
        atom_dim,
        output_dim,
        type_dim=32,
        hidden_dim=None,
        max_pairs=128,
        pooling="sum",
        num_families=len(PHARMACOPHORE_FAMILIES),
        dropout=0.1,
    ):
        super().__init__()
        if pooling not in {"sum", "mean"}:
            raise ValueError("pooling must be 'sum' or 'mean'")
        self.atom_dim = atom_dim
        self.output_dim = output_dim
        self.max_pairs = max_pairs
        self.pooling = pooling
        hidden_dim = hidden_dim or output_dim

        self.type_embedding = nn.Embedding(num_families, type_dim)
        pair_input_dim = atom_dim * 4 + type_dim * 2
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def _cache_to_device(self, pharmacophores, device):
        if not isinstance(pharmacophores, dict):
            return None

        device_key = str(device)
        device_cache = pharmacophores.setdefault("_device_cache", {})
        if device_key not in device_cache:
            device_cache[device_key] = {
                "atom_index": pharmacophores["atom_index"].to(device),
                "atom_owner": pharmacophores["atom_owner"].to(device),
                "atom_count": pharmacophores["atom_count"].to(device),
                "type_ids": pharmacophores["type_ids"].to(device),
                "num_pharmacophores": pharmacophores["num_pharmacophores"],
            }
        return device_cache[device_key]

    def _pool_one_drug(self, atom_embeddings, pharmacophores):
        cached = self._cache_to_device(pharmacophores, atom_embeddings.device)
        if cached is not None:
            num_pharmacophores = cached["num_pharmacophores"]
            if num_pharmacophores == 0:
                empty_nodes = atom_embeddings.new_zeros((0, self.atom_dim))
                empty_types = torch.empty(0, dtype=torch.long, device=atom_embeddings.device)
                return empty_nodes, empty_types

            valid_mask = cached["atom_index"] < atom_embeddings.size(0)
            atom_index = cached["atom_index"][valid_mask]
            atom_owner = cached["atom_owner"][valid_mask]
            if atom_index.numel() == 0:
                empty_nodes = atom_embeddings.new_zeros((0, self.atom_dim))
                empty_types = torch.empty(0, dtype=torch.long, device=atom_embeddings.device)
                return empty_nodes, empty_types

            selected_atoms = atom_embeddings.index_select(0, atom_index)
            node_embeddings = atom_embeddings.new_zeros((num_pharmacophores, self.atom_dim))
            node_embeddings.index_add_(0, atom_owner, selected_atoms)
            if self.pooling == "mean":
                counts = atom_embeddings.new_zeros(num_pharmacophores)
                counts.index_add_(0, atom_owner, torch.ones_like(atom_owner, dtype=atom_embeddings.dtype))
                node_embeddings = node_embeddings / counts.clamp_min(1.0).unsqueeze(-1)
            return node_embeddings, cached["type_ids"]

        node_embeddings = []
        type_ids = []
        num_atoms = atom_embeddings.size(0)
        for feature in pharmacophores:
            atom_ids = [idx for idx in feature["atom_ids"] if idx < num_atoms]
            if not atom_ids:
                continue
            index = torch.tensor(atom_ids, dtype=torch.long, device=atom_embeddings.device)
            pooled = atom_embeddings.index_select(0, index)
            pooled = pooled.sum(dim=0) if self.pooling == "sum" else pooled.mean(dim=0)
            node_embeddings.append(pooled)
            type_ids.append(feature["family_id"])

        if not node_embeddings:
            empty_nodes = atom_embeddings.new_zeros((0, self.atom_dim))
            empty_types = torch.empty(0, dtype=torch.long, device=atom_embeddings.device)
            return empty_nodes, empty_types

        return torch.stack(node_embeddings, dim=0), torch.tensor(
            type_ids,
            dtype=torch.long,
            device=atom_embeddings.device,
        )

    def _encode_one_pair_set(self, drug_a_atoms, drug_a_pharm, drug_b_atoms, drug_b_pharm):
        nodes_a, types_a = self._pool_one_drug(drug_a_atoms, drug_a_pharm)
        nodes_b, types_b = self._pool_one_drug(drug_b_atoms, drug_b_pharm)
        if nodes_a.size(0) == 0 or nodes_b.size(0) == 0:
            return drug_a_atoms.new_zeros((0, self.output_dim))

        pair_left = nodes_a[:, None, :].expand(-1, nodes_b.size(0), -1)
        pair_right = nodes_b[None, :, :].expand(nodes_a.size(0), -1, -1)
        type_left = self.type_embedding(types_a)[:, None, :].expand(-1, nodes_b.size(0), -1)
        type_right = self.type_embedding(types_b)[None, :, :].expand(nodes_a.size(0), -1, -1)

        pair_input = torch.cat(
            [
                pair_left,
                pair_right,
                pair_left * pair_right,
                torch.abs(pair_left - pair_right),
                type_left,
                type_right,
            ],
            dim=-1,
        ).reshape(-1, self.atom_dim * 4 + type_left.size(-1) * 2)

        if pair_input.size(0) > self.max_pairs:
            pair_input = pair_input[: self.max_pairs]
        return self.norm(self.pair_mlp(pair_input))

    def forward(self, drug_a_batch, drug_b_batch, drug_a_features, drug_b_features):
        batch_size = len(drug_a_features)
        encoded_pairs = []
        for batch_idx in range(batch_size):
            start_a = int(drug_a_batch.ptr[batch_idx])
            end_a = int(drug_a_batch.ptr[batch_idx + 1])
            start_b = int(drug_b_batch.ptr[batch_idx])
            end_b = int(drug_b_batch.ptr[batch_idx + 1])
            pairs = self._encode_one_pair_set(
                drug_a_batch.node_representation[start_a:end_a],
                drug_a_features[batch_idx],
                drug_b_batch.node_representation[start_b:end_b],
                drug_b_features[batch_idx],
            )
            encoded_pairs.append(pairs)

        max_len = max([pairs.size(0) for pairs in encoded_pairs] + [1])
        max_len = min(max_len, self.max_pairs)
        output = drug_a_batch.node_representation.new_zeros(
            (batch_size, max_len, self.output_dim)
        )
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=output.device)
        for idx, pairs in enumerate(encoded_pairs):
            length = min(pairs.size(0), max_len)
            if length == 0:
                continue
            output[idx, :length] = pairs[:length]
            mask[idx, :length] = True
        return output, mask
