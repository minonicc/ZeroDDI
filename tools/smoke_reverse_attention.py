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
DDIEGuidedEvidenceSelector = candidate_matching.DDIEGuidedEvidenceSelector
CandidateSpecificDrugPairSelector = candidate_matching.CandidateSpecificDrugPairSelector


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

    # Make candidate-specific selection and padding exclusion deterministic.
    selector = DDIEGuidedEvidenceSelector(
        evidence_dim=2,
        event_dim=2,
        hidden_dim=2,
        use_null_evidence=True,
    )
    with torch.no_grad():
        for projection in (selector.query, selector.key, selector.value):
            projection.weight.copy_(torch.eye(2))
            projection.bias.zero_()
        selector.null_evidence.zero_()
    deterministic_events = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    deterministic_pairs = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0], [100.0, 100.0], [50.0, 50.0]]]
    )
    deterministic_mask = torch.tensor([[True, True, False, False]])
    deterministic_selected, deterministic_attention, deterministic_indices, deterministic_valid = selector(
        deterministic_events,
        deterministic_pairs,
        deterministic_mask,
        top_k=3,
    )
    assert deterministic_indices[0, 0, 0].item() == 0
    assert deterministic_indices[0, 1, 0].item() == 1
    assert deterministic_valid[0, :, :-1].sum(dim=-1).eq(2).all()
    assert deterministic_valid[0, :, -1].all()
    assert not deterministic_valid[0, :, 2].any()
    assert torch.allclose(
        deterministic_attention.sum(dim=-1),
        torch.ones(1, 2),
        atol=1e-6,
    )
    expected_attention = torch.softmax(
        torch.tensor([2.0 / (2.0 ** 0.5), 0.0, 0.0]), dim=0
    )
    # The third real Top-K slot is padding and must receive exactly zero mass;
    # the final slot is the null token. Candidate 0 selects pair 0 before pair
    # 1, while candidate 1 selects them in the opposite order.
    assert torch.allclose(
        deterministic_attention[0, 0],
        torch.tensor(
            [expected_attention[0], expected_attention[1], 0.0, expected_attention[2]]
        ),
        atol=1e-6,
    )
    assert torch.allclose(
        deterministic_attention[0, 1],
        torch.tensor(
            [expected_attention[0], expected_attention[1], 0.0, expected_attention[2]]
        ),
        atol=1e-6,
    )
    expected_selected = torch.tensor(
        [
            [2.0 * expected_attention[0], 2.0 * expected_attention[1]],
            [2.0 * expected_attention[1], 2.0 * expected_attention[0]],
        ]
    )
    assert torch.allclose(deterministic_selected[0], expected_selected, atol=1e-6)
    print("candidate-specific pharmacophore top-k smoke test ok")

    sigmoid_selector = DDIEGuidedEvidenceSelector(
        evidence_dim=2,
        event_dim=2,
        hidden_dim=2,
        use_null_evidence=False,
        top_k_aggregation="sigmoid_mean",
    )
    with torch.no_grad():
        for projection in (
            sigmoid_selector.query,
            sigmoid_selector.key,
            sigmoid_selector.value,
        ):
            projection.weight.copy_(torch.eye(2))
            projection.bias.zero_()
    sigmoid_selected, sigmoid_weights, _, sigmoid_valid = sigmoid_selector(
        deterministic_events,
        deterministic_pairs,
        deterministic_mask,
        top_k=3,
    )
    sigmoid_expected = torch.sigmoid(
        torch.tensor([2.0 / (2.0 ** 0.5), 0.0])
    ) / 2.0
    assert torch.allclose(
        sigmoid_weights[0, 0],
        torch.tensor([sigmoid_expected[0], sigmoid_expected[1], 0.0]),
        atol=1e-6,
    )
    assert sigmoid_valid[0, 0].tolist() == [True, True, False]
    assert torch.allclose(
        sigmoid_selected[0, 0],
        torch.tensor([2.0 * sigmoid_expected[0], 2.0 * sigmoid_expected[1]]),
        atol=1e-6,
    )
    assert sigmoid_weights[0, 0].sum() < 1.0
    print("candidate-specific sigmoid-mean pharmacophore top-k smoke test ok")

    class SharedPairEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.atom_dim = evidence_dim
            self.output_dim = evidence_dim
            self.type_embedding = torch.nn.Embedding(6, 32)
            self.pair_mlp = torch.nn.Sequential(
                torch.nn.Linear(evidence_dim * 4 + 64, evidence_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(0.0),
                torch.nn.Linear(evidence_dim, evidence_dim),
            )
            self.norm = torch.nn.LayerNorm(evidence_dim)

    shared_pair_encoder = SharedPairEncoder()
    drug_top_k_model = ReverseAttentionCandidateMatcher(
        pair_dim=pair_dim,
        evidence_dim=evidence_dim,
        event_dim=event_dim,
        hidden_dim=hidden_dim,
        use_pharmacophore_evidence=True,
        pharmacophore_evidence_dim=evidence_dim,
        pharmacophore_drug_top_k=3,
        pharmacophore_candidate_chunk_size=2,
        pharmacophore_shared_pair_encoder=shared_pair_encoder,
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
    drug_b_count = drug_b_mask.sum(dim=-1)
    precomputed_pairs = torch.randn(batch_size, 20, evidence_dim)
    precomputed_pair_mask = torch.zeros(batch_size, 20, dtype=torch.bool)
    for batch_index in range(batch_size):
        pair_count = int(drug_a_mask[batch_index].sum() * drug_b_count[batch_index])
        precomputed_pair_mask[batch_index, :pair_count] = True
    drug_top_k_outputs = drug_top_k_model(
        pair_repr,
        evidence_tokens,
        event_tokens,
        labels,
        pharmacophore_evidence_tokens=precomputed_pairs,
        pharmacophore_evidence_mask=precomputed_pair_mask,
        pharmacophore_drug_a_nodes=drug_a_nodes,
        pharmacophore_drug_a_types=drug_a_types,
        pharmacophore_drug_a_mask=drug_a_mask,
        pharmacophore_drug_b_nodes=drug_b_nodes,
        pharmacophore_drug_b_types=drug_b_types,
        pharmacophore_drug_b_mask=drug_b_mask,
        pharmacophore_drug_b_count=drug_b_count,
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

    # The production D3 path selects nodes before constructing pairs. It does
    # not materialize the complete Cartesian product, but still reports the
    # original available-pair count for coverage diagnostics.
    available_pair_count = drug_a_mask.sum(dim=-1) * drug_b_mask.sum(dim=-1)
    direct_drug_top_k_outputs = drug_top_k_model(
        pair_repr,
        evidence_tokens,
        event_tokens,
        labels,
        pharmacophore_valid_pair_count=available_pair_count,
        pharmacophore_drug_a_nodes=drug_a_nodes,
        pharmacophore_drug_a_types=drug_a_types,
        pharmacophore_drug_a_mask=drug_a_mask,
        pharmacophore_drug_b_nodes=drug_b_nodes,
        pharmacophore_drug_b_types=drug_b_types,
        pharmacophore_drug_b_mask=drug_b_mask,
        pharmacophore_drug_b_count=drug_b_count,
    )
    direct_selection = direct_drug_top_k_outputs["pharmacophore_drug_selection"]
    assert direct_selection["pair_mask"].shape == (batch_size, num_events, 10)
    assert torch.equal(
        direct_drug_top_k_outputs["pharmacophore_valid_pair_count"],
        available_pair_count,
    )
    assert torch.allclose(
        direct_drug_top_k_outputs["pharmacophore_attention"].sum(dim=-1),
        torch.ones(batch_size, num_events),
        atol=1e-6,
    )
    direct_drug_top_k_outputs["loss"].backward()
    assert shared_pair_encoder.pair_mlp[0].weight.grad is not None
    selector_parameter_names = dict(
        drug_top_k_model.pharmacophore_drug_selector.named_parameters()
    )
    assert not any(name.startswith("pair_mlp.") for name in selector_parameter_names)
    assert (
        drug_top_k_model.pharmacophore_drug_selector._shared_pair_encoder
        is shared_pair_encoder
    )

    class SharedD3Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left_pair_encoder = SharedPairEncoder()
            self.matcher = ReverseAttentionCandidateMatcher(
                pair_dim=pair_dim,
                evidence_dim=evidence_dim,
                event_dim=event_dim,
                hidden_dim=hidden_dim,
                use_pharmacophore_evidence=True,
                pharmacophore_evidence_dim=evidence_dim,
                pharmacophore_drug_top_k=3,
                pharmacophore_shared_pair_encoder=self.left_pair_encoder,
            )

    wrapper = SharedD3Wrapper()
    state = wrapper.state_dict()
    assert any(name.startswith("left_pair_encoder.pair_mlp.") for name in state)
    assert not any(
        "matcher.pharmacophore_drug_selector.pair_mlp" in name for name in state
    )
    restored = SharedD3Wrapper()
    incompatible = restored.load_state_dict(state)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    assert (
        restored.matcher.pharmacophore_drug_selector._shared_pair_encoder
        is restored.left_pair_encoder
    )

    deterministic_drug_selector = CandidateSpecificDrugPairSelector(
        atom_dim=2,
        event_dim=2,
        pair_dim=2,
        output_dim=2,
        top_k=1,
        use_null_evidence=True,
    )
    with torch.no_grad():
        deterministic_drug_selector.query.weight.copy_(torch.eye(2))
        deterministic_drug_selector.query.bias.zero_()
        deterministic_drug_selector.node_key.weight.copy_(torch.eye(2))
        deterministic_drug_selector.node_key.bias.zero_()
        deterministic_drug_selector.pair_key.weight.zero_()
        deterministic_drug_selector.pair_key.bias.zero_()
        deterministic_drug_selector.pair_value.weight.copy_(torch.eye(2))
        deterministic_drug_selector.pair_value.bias.zero_()
        deterministic_drug_selector.null_pair.zero_()
    deterministic_nodes = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0], [100.0, 100.0]]]
    )
    deterministic_node_mask = torch.tensor([[True, True, False]])
    deterministic_types = torch.zeros(1, 3, dtype=torch.long)
    row_major_pairs = torch.tensor(
        [[[10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]]]
    )
    deterministic_drug_outputs = deterministic_drug_selector(
        deterministic_events,
        deterministic_nodes,
        deterministic_types,
        deterministic_node_mask,
        deterministic_nodes,
        deterministic_types,
        deterministic_node_mask,
        precomputed_pair_tokens=row_major_pairs,
        precomputed_pair_mask=torch.ones(1, 4, dtype=torch.bool),
        drug_b_count=torch.tensor([2]),
    )
    assert deterministic_drug_outputs["indices_a"][0, 0, 0].item() == 0
    assert deterministic_drug_outputs["indices_a"][0, 1, 0].item() == 1
    assert deterministic_drug_outputs["indices_b"][0, 0, 0].item() == 0
    assert deterministic_drug_outputs["indices_b"][0, 1, 0].item() == 1
    assert deterministic_drug_outputs["valid_a"].all()
    assert deterministic_drug_outputs["valid_b"].all()
    # With one real pair plus a zero null token at equal score, the selected
    # value is exactly half of row-major pair 0 for candidate 0 and pair 3 for 1.
    assert torch.allclose(
        deterministic_drug_outputs["selected"],
        torch.tensor([[[5.0, 0.0], [20.0, 0.0]]]),
        atol=1e-6,
    )
    print("candidate-specific per-drug pharmacophore top-k smoke test ok")


if __name__ == "__main__":
    main()
