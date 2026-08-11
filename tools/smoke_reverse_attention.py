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

    num_nodes = 11
    num_edges = 24
    graph_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)
    graph = {
        "node_ids": torch.randint(1, 16, (batch_size, num_nodes)),
        "node_types": torch.randint(1, 6, (batch_size, num_nodes)),
        "distance_to_a": torch.randint(1, 6, (batch_size, num_nodes)),
        "distance_to_b": torch.randint(1, 6, (batch_size, num_nodes)),
        "node_mask": graph_mask,
        "edge_index": torch.randint(0, num_nodes, (batch_size, 2, num_edges)),
        "edge_relations": torch.randint(1, 10, (batch_size, num_edges)),
        "edge_mask": torch.ones(batch_size, num_edges, dtype=torch.bool),
    }
    kg_graph_model = ReverseAttentionCandidateMatcher(
        pair_dim=pair_dim,
        evidence_dim=evidence_dim,
        event_dim=event_dim,
        hidden_dim=hidden_dim,
        use_kg_evidence=True,
        kg_evidence_dim=evidence_dim,
        kg_hidden_dim=hidden_dim,
        kg_feature_vocab_sizes={
            "entity": 16,
            "type": 6,
            "relation": 10,
            "distance": 6,
        },
        kg_graph_evidence=True,
    )
    kg_graph_outputs = kg_graph_model(
        pair_repr,
        evidence_tokens,
        event_tokens,
        labels=labels,
        kg_evidence_tokens=graph,
        kg_evidence_mask=graph_mask,
    )
    kg_graph_outputs["loss"].backward()
    print("kg edge-aware GraphSAGE smoke test ok")
    print("kg_graph_attention:", tuple(kg_graph_outputs["kg_attention"].shape))


if __name__ == "__main__":
    main()
