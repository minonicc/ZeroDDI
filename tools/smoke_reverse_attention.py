import os
import sys
import importlib.util

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODULE_PATH = os.path.join(ROOT, "models", "classifier", "candidate_matching.py")
spec = importlib.util.spec_from_file_location("candidate_matching", MODULE_PATH)
candidate_matching = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate_matching)
ReverseAttentionCandidateMatcher = candidate_matching.ReverseAttentionCandidateMatcher


def main():
    torch.manual_seed(42)

    batch_size = 4
    num_events = 7
    event_len = 12
    num_evidence = 16
    pair_dim = 256
    evidence_dim = 300
    event_dim = 256
    hidden_dim = 256

    pair_repr = torch.randn(batch_size, pair_dim)
    evidence_tokens = torch.randn(batch_size, num_evidence, evidence_dim)
    event_tokens = torch.randn(num_events, event_len, event_dim)
    labels = torch.tensor([0, 2, 4, 6], dtype=torch.long)

    model = ReverseAttentionCandidateMatcher(
        pair_dim=pair_dim,
        evidence_dim=evidence_dim,
        event_dim=event_dim,
        hidden_dim=hidden_dim,
    )

    outputs = model(pair_repr, evidence_tokens, event_tokens, labels=labels)
    outputs["loss"].backward()

    print("reverse attention smoke test ok")
    print("logits:", tuple(outputs["logits"].shape))
    print("attention:", tuple(outputs["attention"].shape))
    print("selected_evidence:", tuple(outputs["selected_evidence"].shape))
    print("loss:", round(outputs["loss"].item(), 6))

    kg_tokens = torch.randn(batch_size, 9, evidence_dim)
    kg_mask = torch.ones(batch_size, 9, dtype=torch.bool)
    kg_model = ReverseAttentionCandidateMatcher(
        pair_dim=pair_dim,
        evidence_dim=evidence_dim,
        event_dim=event_dim,
        hidden_dim=hidden_dim,
        use_kg_evidence=True,
        kg_evidence_dim=evidence_dim,
        kg_hidden_dim=hidden_dim,
    )
    kg_outputs = kg_model(
        pair_repr,
        evidence_tokens,
        event_tokens,
        labels=labels,
        kg_evidence_tokens=kg_tokens,
        kg_evidence_mask=kg_mask,
    )
    kg_outputs["loss"].backward()

    fallback_outputs = kg_model(pair_repr, evidence_tokens, event_tokens, labels=labels)

    print("kg reverse attention smoke test ok")
    print("kg_attention:", tuple(kg_outputs["kg_attention"].shape))
    print("kg_selected_evidence:", tuple(kg_outputs["kg_selected_evidence"].shape))
    print("fallback_logits:", tuple(fallback_outputs["logits"].shape))

    kg_feature_tokens = torch.randint(1, 8, (batch_size, 9, 5), dtype=torch.long)
    kg_feature_model = ReverseAttentionCandidateMatcher(
        pair_dim=pair_dim,
        evidence_dim=evidence_dim,
        event_dim=event_dim,
        hidden_dim=hidden_dim,
        use_kg_evidence=True,
        kg_evidence_dim=evidence_dim,
        kg_hidden_dim=hidden_dim,
        kg_feature_vocab_sizes={
            "entity": 16,
            "type": 8,
            "relation": 8,
            "side": 8,
            "distance": 8,
        },
    )
    kg_feature_outputs = kg_feature_model(
        pair_repr,
        evidence_tokens,
        event_tokens,
        labels=labels,
        kg_evidence_tokens=kg_feature_tokens,
        kg_evidence_mask=kg_mask,
    )
    kg_feature_outputs["loss"].backward()
    print("kg feature token smoke test ok")

    pharmacophore_tokens = torch.randn(batch_size, 11, evidence_dim)
    pharmacophore_mask = torch.ones(batch_size, 11, dtype=torch.bool)
    pharmacophore_mask[0, 7:] = False
    pharmacophore_mask[1, :] = False
    pharmacophore_gate_model = ReverseAttentionCandidateMatcher(
        pair_dim=pair_dim,
        evidence_dim=evidence_dim,
        event_dim=event_dim,
        hidden_dim=hidden_dim,
        use_pharmacophore_evidence=True,
        pharmacophore_evidence_dim=evidence_dim,
        pharmacophore_hidden_dim=hidden_dim,
        use_pharmacophore_gate=True,
    )
    pharmacophore_outputs = pharmacophore_gate_model(
        pair_repr,
        evidence_tokens,
        event_tokens,
        labels=labels,
        pharmacophore_evidence_tokens=pharmacophore_tokens,
        pharmacophore_evidence_mask=pharmacophore_mask,
    )
    pharmacophore_outputs["loss"].backward()

    pharmacophore_gate = pharmacophore_outputs["pharmacophore_gate"]
    assert pharmacophore_gate.shape == (batch_size, num_events, 1)
    assert torch.all((pharmacophore_gate >= 0) & (pharmacophore_gate <= 1))
    # The existing null token is intentionally retained as the final position.
    assert pharmacophore_outputs["pharmacophore_attention"].shape == (
        batch_size,
        num_events,
        12,
    )
    assert torch.allclose(
        pharmacophore_outputs["pharmacophore_attention"].sum(dim=-1),
        torch.ones(batch_size, num_events),
        atol=1e-6,
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in pharmacophore_gate_model.pharmacophore_gate.parameters()
    )
    print("pharmacophore gate smoke test ok")

    no_substructure_model = ReverseAttentionCandidateMatcher(
        pair_dim=pair_dim,
        evidence_dim=evidence_dim,
        event_dim=event_dim,
        hidden_dim=hidden_dim,
        use_substructure_evidence=False,
        use_kg_evidence=True,
        kg_evidence_dim=evidence_dim,
        use_pharmacophore_evidence=True,
        pharmacophore_evidence_dim=evidence_dim,
    )
    no_substructure_outputs = no_substructure_model(
        pair_repr,
        None,
        event_tokens,
        labels,
        kg_evidence_tokens=kg_tokens,
        kg_evidence_mask=kg_mask,
        pharmacophore_evidence_tokens=pharmacophore_tokens,
        pharmacophore_evidence_mask=pharmacophore_mask,
    )
    no_substructure_outputs["loss"].backward()
    assert no_substructure_outputs["attention"] is None
    assert no_substructure_outputs["logits"].shape == (batch_size, num_events)
    print("no substructure query smoke test ok")

    top_k_model = ReverseAttentionCandidateMatcher(
        pair_dim=pair_dim,
        evidence_dim=evidence_dim,
        event_dim=event_dim,
        hidden_dim=hidden_dim,
        use_pharmacophore_evidence=True,
        pharmacophore_evidence_dim=evidence_dim,
        pharmacophore_top_k=5,
    )
    top_k_outputs = top_k_model(
        pair_repr,
        evidence_tokens,
        event_tokens,
        labels,
        pharmacophore_evidence_tokens=pharmacophore_tokens,
        pharmacophore_evidence_mask=pharmacophore_mask,
    )
    top_k_outputs["loss"].backward()
    top_k_indices = top_k_outputs["pharmacophore_selection_indices"]
    top_k_mask = top_k_outputs["pharmacophore_selection_mask"]
    assert top_k_indices.shape == (batch_size, num_events, 5)
    assert top_k_mask.shape == (batch_size, num_events, 6)  # five real + null
    assert not top_k_mask[1, :, :-1].any()
    assert top_k_mask[1, :, -1].all()
    assert torch.allclose(
        top_k_outputs["pharmacophore_attention"].sum(dim=-1),
        torch.ones(batch_size, num_events),
        atol=1e-6,
    )
    assert torch.all(
        top_k_indices[0][top_k_mask[0, :, :-1]] < 7
    )
    print("candidate-specific pharmacophore top-k smoke test ok")

    drug_top_k_model = ReverseAttentionCandidateMatcher(
        pair_dim=pair_dim,
        evidence_dim=evidence_dim,
        event_dim=event_dim,
        hidden_dim=hidden_dim,
        use_pharmacophore_evidence=True,
        pharmacophore_evidence_dim=evidence_dim,
        pharmacophore_drug_top_k=3,
        pharmacophore_candidate_chunk_size=2,
    )
    drug_a_nodes = torch.randn(batch_size, 5, evidence_dim)
    drug_b_nodes = torch.randn(batch_size, 4, evidence_dim)
    drug_a_types = torch.randint(0, 6, (batch_size, 5))
    drug_b_types = torch.randint(0, 6, (batch_size, 4))
    drug_a_mask = torch.ones(batch_size, 5, dtype=torch.bool)
    drug_b_mask = torch.ones(batch_size, 4, dtype=torch.bool)
    drug_a_mask[0, 2:] = False
    drug_b_mask[0, 1:] = False
    drug_a_mask[1, :] = False
    drug_b_mask[1, :] = False
    drug_top_k_outputs = drug_top_k_model(
        pair_repr,
        evidence_tokens,
        event_tokens,
        labels,
        pharmacophore_drug_a_nodes=drug_a_nodes,
        pharmacophore_drug_a_types=drug_a_types,
        pharmacophore_drug_a_mask=drug_a_mask,
        pharmacophore_drug_b_nodes=drug_b_nodes,
        pharmacophore_drug_b_types=drug_b_types,
        pharmacophore_drug_b_mask=drug_b_mask,
    )
    drug_top_k_outputs["loss"].backward()
    drug_selection = drug_top_k_outputs["pharmacophore_drug_selection"]
    assert drug_selection["indices_a"].shape == (batch_size, num_events, 3)
    assert drug_selection["indices_b"].shape == (batch_size, num_events, 3)
    assert drug_selection["pair_mask"].shape == (batch_size, num_events, 10)
    assert drug_selection["pair_mask"][0, :, :-1].sum(dim=-1).eq(2).all()
    assert not drug_selection["pair_mask"][1, :, :-1].any()
    assert drug_selection["pair_mask"][:, :, -1].all()
    assert torch.allclose(
        drug_top_k_outputs["pharmacophore_attention"].sum(dim=-1),
        torch.ones(batch_size, num_events),
        atol=1e-6,
    )
    print("candidate-specific per-drug pharmacophore top-k smoke test ok")


if __name__ == "__main__":
    main()
