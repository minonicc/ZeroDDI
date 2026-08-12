"""Smoke-test pharmacophore pooling, pairing, truncation, and type alignment."""

from pathlib import Path
from types import SimpleNamespace
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.left.graph.pharmacophore import PharmacophorePairEncoder  # noqa: E402


def cached_features(atom_index, atom_owner, atom_count, type_ids):
    return {
        "atom_index": torch.tensor(atom_index, dtype=torch.long),
        "atom_owner": torch.tensor(atom_owner, dtype=torch.long),
        "atom_count": torch.tensor(atom_count, dtype=torch.long),
        "type_ids": torch.tensor(type_ids, dtype=torch.long),
        "num_pharmacophores": len(type_ids),
    }


def make_inputs():
    drug_a = SimpleNamespace(
        ptr=torch.tensor([0, 3, 4]),
        node_representation=torch.tensor(
            [[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [4.0, 4.0]],
            requires_grad=True,
        ),
    )
    drug_b = SimpleNamespace(
        ptr=torch.tensor([0, 2, 3]),
        node_representation=torch.tensor(
            [[0.0, 1.0], [0.0, 3.0], [5.0, 5.0]],
            requires_grad=True,
        ),
    )
    features_a = [
        cached_features([0, 1, 2], [0, 0, 1], [2, 1], [0, 2]),
        cached_features([], [], [], []),
    ]
    features_b = [
        cached_features([0, 1], [0, 1], [1, 1], [1, 3]),
        cached_features([0], [0], [1], [4]),
    ]
    return drug_a, drug_b, features_a, features_b


def run_case(pooling, batch_pair_mlp):
    drug_a, drug_b, features_a, features_b = make_inputs()
    encoder = PharmacophorePairEncoder(
        atom_dim=2,
        output_dim=4,
        hidden_dim=4,
        type_dim=2,
        max_pairs=3,
        pooling=pooling,
        dropout=0.0,
        batch_pair_mlp=batch_pair_mlp,
    )
    tokens, mask = encoder(drug_a, drug_b, features_a, features_b)
    pair_types = encoder.pair_type_ids(
        features_a, features_b, tokens.size(1), tokens.device
    )
    assert tokens.shape == (2, 3, 4)
    assert mask.tolist() == [[True, True, True], [False, False, False]]
    # Row-major products: (0,1), (0,3), (2,1); the fourth pair is truncated.
    assert pair_types.tolist() == [[1, 3, 13], [-1, -1, -1]]
    assert torch.isfinite(tokens).all()
    tokens[mask].sum().backward()
    assert drug_a.node_representation.grad is not None
    assert drug_b.node_representation.grad is not None


def run_batched_equivalence(pooling):
    reference = PharmacophorePairEncoder(
        atom_dim=2,
        output_dim=4,
        hidden_dim=4,
        type_dim=2,
        max_pairs=3,
        pooling=pooling,
        dropout=0.0,
        batch_pair_mlp=False,
    )
    batched = PharmacophorePairEncoder(
        atom_dim=2,
        output_dim=4,
        hidden_dim=4,
        type_dim=2,
        max_pairs=3,
        pooling=pooling,
        dropout=0.0,
        batch_pair_mlp=True,
    )
    batched.load_state_dict(reference.state_dict())
    reference.eval()
    batched.eval()
    inputs = make_inputs()
    reference_tokens, reference_mask = reference(*inputs)
    batched_tokens, batched_mask = batched(*inputs)
    assert torch.equal(reference_mask, batched_mask)
    assert torch.allclose(reference_tokens, batched_tokens, atol=1e-7)


def main():
    for pooling in ("sum", "mean"):
        for batch_pair_mlp in (False, True):
            run_case(pooling, batch_pair_mlp)
        run_batched_equivalence(pooling)
    print("pharmacophore pair encoder smoke test ok")


if __name__ == "__main__":
    main()
