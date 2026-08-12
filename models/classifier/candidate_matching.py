import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def pool_event_tokens(event_tokens):
    if event_tokens.dim() == 3:
        return event_tokens.mean(dim=1)
    if event_tokens.dim() == 2:
        return event_tokens
    raise ValueError("event_tokens must be [C, L, D] or [C, D]")


class DDIEGuidedEvidenceSelector(nn.Module):
    """Select drug evidence tokens for each candidate DDIE representation."""

    def __init__(self, evidence_dim, event_dim, hidden_dim, use_null_evidence=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_null_evidence = use_null_evidence
        self.query = nn.Linear(event_dim, hidden_dim)
        self.key = nn.Linear(evidence_dim, hidden_dim)
        self.value = nn.Linear(evidence_dim, hidden_dim)
        if self.use_null_evidence:
            self.null_evidence = nn.Parameter(torch.zeros(1, 1, evidence_dim))

    def forward(self, event_tokens, evidence_tokens, evidence_mask=None, top_k=None):
        """
        Args:
            event_tokens: [num_events, event_len, event_dim] or [num_events, event_dim]
            evidence_tokens: [batch, num_evidence, evidence_dim]
            evidence_mask: optional bool tensor, [batch, num_evidence], True for valid tokens

        Returns:
            selected: [batch, num_events, hidden_dim]
            attention: [batch, num_events, num_evidence]
        """
        event_repr = pool_event_tokens(event_tokens)

        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            return self._forward_top_k(
                event_repr, evidence_tokens, evidence_mask, top_k
            )

        if self.use_null_evidence:
            null_evidence = self.null_evidence.expand(evidence_tokens.size(0), -1, -1)
            evidence_tokens = torch.cat([evidence_tokens, null_evidence], dim=1)
            if evidence_mask is not None:
                null_mask = torch.ones(
                    evidence_mask.size(0),
                    1,
                    dtype=torch.bool,
                    device=evidence_mask.device,
                )
                evidence_mask = torch.cat([evidence_mask, null_mask], dim=1)

        query = self.query(event_repr).unsqueeze(0)  # [1, C, H]
        key = self.key(evidence_tokens)  # [B, E, H]
        value = self.value(evidence_tokens)  # [B, E, H]

        scores = torch.matmul(query.expand(key.size(0), -1, -1), key.transpose(1, 2))
        scores = scores / math.sqrt(self.hidden_dim)

        if evidence_mask is not None:
            scores = scores.masked_fill(~evidence_mask.unsqueeze(1), torch.finfo(scores.dtype).min)

        attention = F.softmax(scores, dim=-1)
        selected = torch.matmul(attention, value)
        return selected, attention, None, evidence_mask

    def _forward_top_k(self, event_repr, evidence_tokens, evidence_mask, top_k):
        """Select candidate-specific real evidence before adding the null token."""
        batch_size, num_evidence, _ = evidence_tokens.shape
        if evidence_mask is None:
            evidence_mask = torch.ones(
                batch_size,
                num_evidence,
                dtype=torch.bool,
                device=evidence_tokens.device,
            )

        query = self.query(event_repr).unsqueeze(0).expand(batch_size, -1, -1)
        key = self.key(evidence_tokens)
        value = self.value(evidence_tokens)
        selected_count = min(int(top_k), num_evidence)
        if self.use_null_evidence:
            null = self.null_evidence.expand(batch_size, -1, -1)
            null_key = self.key(null)
            null_value = self.value(null)

        selected_chunks = []
        attention_chunks = []
        index_chunks = []
        valid_chunks = []
        candidate_chunk_size = 16
        for start in range(0, event_repr.size(0), candidate_chunk_size):
            end = min(start + candidate_chunk_size, event_repr.size(0))
            chunk_query = query[:, start:end]
            scores = torch.matmul(chunk_query, key.transpose(1, 2))
            scores = scores / math.sqrt(self.hidden_dim)
            scores = scores.masked_fill(
                ~evidence_mask.unsqueeze(1), torch.finfo(scores.dtype).min
            )
            top_scores, top_indices = torch.topk(scores, selected_count, dim=-1)
            expanded_mask = evidence_mask.unsqueeze(1).expand(-1, end - start, -1)
            top_valid = torch.gather(expanded_mask, 2, top_indices)
            expanded_value = value.unsqueeze(1).expand(-1, end - start, -1, -1)
            top_values = torch.gather(
                expanded_value,
                2,
                top_indices.unsqueeze(-1).expand(-1, -1, -1, value.size(-1)),
            )
            if self.use_null_evidence:
                null_scores = torch.matmul(chunk_query, null_key.transpose(1, 2))
                null_scores = null_scores / math.sqrt(self.hidden_dim)
                top_scores = torch.cat([top_scores, null_scores], dim=-1)
                top_values = torch.cat(
                    [
                        top_values,
                        null_value.unsqueeze(1).expand(-1, end - start, -1, -1),
                    ],
                    dim=2,
                )
                top_valid = torch.cat(
                    [
                        top_valid,
                        torch.ones(
                            batch_size,
                            end - start,
                            1,
                            dtype=torch.bool,
                            device=evidence_tokens.device,
                        ),
                    ],
                    dim=-1,
                )
            top_scores = top_scores.masked_fill(
                ~top_valid, torch.finfo(top_scores.dtype).min
            )
            attention = F.softmax(top_scores, dim=-1).masked_fill(~top_valid, 0.0)
            selected = torch.matmul(attention.unsqueeze(-2), top_values).squeeze(-2)
            selected_chunks.append(selected)
            attention_chunks.append(attention)
            index_chunks.append(top_indices)
            valid_chunks.append(top_valid)
        return (
            torch.cat(selected_chunks, dim=1),
            torch.cat(attention_chunks, dim=1),
            torch.cat(index_chunks, dim=1),
            torch.cat(valid_chunks, dim=1),
        )


class KGEvidenceFeatureEncoder(nn.Module):
    """Embed categorical KG evidence features into dense KG evidence tokens."""

    def __init__(
        self,
        vocab_sizes,
        output_dim,
        padding_idx=0,
        dropout=0.1,
    ):
        super().__init__()
        self.feature_names = ("entity", "type", "relation", "side", "distance")
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(vocab_sizes[name], output_dim, padding_idx=padding_idx)
                for name in self.feature_names
            ]
        )
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        if features.size(-1) != len(self.feature_names):
            raise ValueError(
                f"KG evidence features must have {len(self.feature_names)} fields, "
                f"got {features.size(-1)}"
            )
        encoded = 0
        for index, embedding in enumerate(self.embeddings):
            encoded = encoded + embedding(features[..., index].long())
        return self.dropout(self.norm(encoded))


class CandidateSpecificDrugPairSelector(nn.Module):
    """Select each drug's pharmacophores per DDIE, then encode their product."""

    def __init__(
        self,
        atom_dim,
        event_dim,
        pair_dim,
        output_dim,
        top_k,
        type_dim=32,
        num_types=6,
        dropout=0.1,
        candidate_chunk_size=8,
        use_null_evidence=True,
        shared_pair_encoder=None,
    ):
        super().__init__()
        self.top_k = int(top_k)
        self.output_dim = output_dim
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.use_null_evidence = use_null_evidence
        self.query = nn.Linear(event_dim, output_dim)
        self.node_key = nn.Linear(atom_dim, output_dim)
        # D3 must change selection, not silently replace the baseline pair
        # encoder. Keep a non-registered reference because the shared module is
        # already registered under Leftmodel and should appear once in the
        # optimizer/state dict.
        object.__setattr__(self, "_shared_pair_encoder", shared_pair_encoder)
        if shared_pair_encoder is None:
            self.type_embedding = nn.Embedding(num_types, type_dim)
            self.pair_mlp = nn.Sequential(
                nn.Linear(atom_dim * 4 + type_dim * 2, pair_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(pair_dim, pair_dim),
            )
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            if shared_pair_encoder.atom_dim != atom_dim:
                raise ValueError("shared pair encoder atom dimension mismatch")
            if shared_pair_encoder.output_dim != pair_dim:
                raise ValueError("shared pair encoder output dimension mismatch")
        self.pair_key = nn.Linear(pair_dim, output_dim)
        self.pair_value = nn.Linear(pair_dim, output_dim)
        if self.use_null_evidence:
            self.null_pair = nn.Parameter(torch.zeros(1, 1, pair_dim))

    def _pair_modules(self):
        shared = self._shared_pair_encoder
        if shared is not None:
            return shared.type_embedding, shared.pair_mlp, shared.norm
        return self.type_embedding, self.pair_mlp, self.pair_norm

    def _top_nodes(self, query, nodes, mask):
        scores = torch.matmul(query, self.node_key(nodes).transpose(1, 2))
        scores = scores / math.sqrt(self.output_dim)
        scores = scores.masked_fill(
            ~mask.unsqueeze(1), torch.finfo(scores.dtype).min
        )
        count = min(self.top_k, nodes.size(1))
        _, indices = torch.topk(scores, count, dim=-1)
        valid = torch.gather(
            mask.unsqueeze(1).expand(-1, query.size(1), -1), 2, indices
        )
        return indices, valid

    @staticmethod
    def _gather_candidate(values, indices):
        expanded = values.unsqueeze(1).expand(-1, indices.size(1), -1, *values.shape[2:])
        gather_index = indices
        for _ in values.shape[2:]:
            gather_index = gather_index.unsqueeze(-1)
        gather_index = gather_index.expand(
            *indices.shape, *values.shape[2:]
        )
        return torch.gather(expanded, 2, gather_index)

    def _encode_chunk(self, chunk_query, left, right, left_types, right_types, pair_valid):
        type_embedding, pair_mlp, pair_norm = self._pair_modules()
        left_type = type_embedding(left_types)
        right_type = type_embedding(right_types)
        pair_left = left.unsqueeze(3).expand(-1, -1, -1, right.size(2), -1)
        pair_right = right.unsqueeze(2).expand(-1, -1, left.size(2), -1, -1)
        type_left = left_type.unsqueeze(3).expand(-1, -1, -1, right.size(2), -1)
        type_right = right_type.unsqueeze(2).expand(-1, -1, left.size(2), -1, -1)
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
        )
        pair_shape = pair_input.shape[:-1]
        pair_tokens = pair_norm(pair_mlp(pair_input)).reshape(
            pair_shape[0], pair_shape[1], -1, self.pair_key.in_features
        )
        pair_scores = (chunk_query.unsqueeze(2) * self.pair_key(pair_tokens)).sum(dim=-1)
        pair_scores = pair_scores / math.sqrt(self.output_dim)
        pair_values = self.pair_value(pair_tokens)
        if self.use_null_evidence:
            null_pair = self.null_pair.expand(left.size(0), -1, -1)
            null_key = self.pair_key(null_pair).unsqueeze(1)
            null_value = self.pair_value(null_pair).unsqueeze(1).expand(
                -1, chunk_query.size(1), -1, -1
            )
            null_score = (chunk_query.unsqueeze(2) * null_key).sum(dim=-1)
            null_score = null_score / math.sqrt(self.output_dim)
            pair_scores = torch.cat([pair_scores, null_score], dim=-1)
            pair_values = torch.cat([pair_values, null_value], dim=2)
        pair_scores = pair_scores.masked_fill(
            ~pair_valid, torch.finfo(pair_scores.dtype).min
        )
        attention = F.softmax(pair_scores, dim=-1).masked_fill(~pair_valid, 0.0)
        selected = torch.matmul(attention.unsqueeze(-2), pair_values).squeeze(-2)
        return selected, attention

    def _aggregate_precomputed_chunk(
        self, chunk_query, pair_tokens, linear_indices, pair_valid
    ):
        expanded_pairs = pair_tokens.unsqueeze(1).expand(
            -1, linear_indices.size(1), -1, -1
        )
        selected_pairs = torch.gather(
            expanded_pairs,
            2,
            linear_indices.unsqueeze(-1).expand(
                -1, -1, -1, pair_tokens.size(-1)
            ),
        )
        pair_scores = (
            chunk_query.unsqueeze(2) * self.pair_key(selected_pairs)
        ).sum(dim=-1) / math.sqrt(self.output_dim)
        pair_values = self.pair_value(selected_pairs)
        if self.use_null_evidence:
            null_pair = self.null_pair.expand(pair_tokens.size(0), -1, -1)
            null_score = (
                chunk_query.unsqueeze(2) * self.pair_key(null_pair).unsqueeze(1)
            ).sum(dim=-1) / math.sqrt(self.output_dim)
            null_value = self.pair_value(null_pair).unsqueeze(1).expand(
                -1, chunk_query.size(1), -1, -1
            )
            pair_scores = torch.cat([pair_scores, null_score], dim=-1)
            pair_values = torch.cat([pair_values, null_value], dim=2)
        pair_scores = pair_scores.masked_fill(
            ~pair_valid, torch.finfo(pair_scores.dtype).min
        )
        attention = F.softmax(pair_scores, dim=-1).masked_fill(~pair_valid, 0.0)
        selected = torch.matmul(attention.unsqueeze(-2), pair_values).squeeze(-2)
        return selected, attention

    def forward(
        self,
        event_tokens,
        nodes_a,
        types_a,
        mask_a,
        nodes_b,
        types_b,
        mask_b,
        precomputed_pair_tokens=None,
        precomputed_pair_mask=None,
        drug_b_count=None,
    ):
        event_repr = pool_event_tokens(event_tokens)
        query = self.query(event_repr).unsqueeze(0).expand(nodes_a.size(0), -1, -1)
        indices_a, valid_a = self._top_nodes(query, nodes_a, mask_a)
        indices_b, valid_b = self._top_nodes(query, nodes_b, mask_b)
        if precomputed_pair_tokens is not None:
            if precomputed_pair_mask is None or drug_b_count is None:
                raise ValueError(
                    "precomputed pair tokens require pair mask and drug B counts"
                )
            return self._forward_precomputed(
                query,
                indices_a,
                valid_a,
                indices_b,
                valid_b,
                types_a,
                mask_a,
                types_b,
                mask_b,
                precomputed_pair_tokens,
                precomputed_pair_mask,
                drug_b_count,
            )
        selected_a = self._gather_candidate(nodes_a, indices_a)
        selected_b = self._gather_candidate(nodes_b, indices_b)
        selected_types_a = self._gather_candidate(types_a.unsqueeze(-1), indices_a).squeeze(-1)
        selected_types_b = self._gather_candidate(types_b.unsqueeze(-1), indices_b).squeeze(-1)

        selected_chunks = []
        attention_chunks = []
        pair_mask_chunks = []
        num_candidates = event_repr.size(0)
        for start in range(0, num_candidates, self.candidate_chunk_size):
            end = min(start + self.candidate_chunk_size, num_candidates)
            left = selected_a[:, start:end]
            right = selected_b[:, start:end]
            pair_valid = (
                valid_a[:, start:end].unsqueeze(-1)
                & valid_b[:, start:end].unsqueeze(-2)
            ).reshape(left.size(0), left.size(1), -1)
            if self.use_null_evidence:
                pair_valid = torch.cat(
                    [
                        pair_valid,
                        torch.ones(
                            nodes_a.size(0),
                            end - start,
                            1,
                            dtype=torch.bool,
                            device=nodes_a.device,
                        ),
                    ],
                    dim=-1,
                )
            encode_args = (
                query[:, start:end],
                left,
                right,
                selected_types_a[:, start:end],
                selected_types_b[:, start:end],
                pair_valid,
            )
            if self.training:
                selected, attention = checkpoint(self._encode_chunk, *encode_args)
            else:
                selected, attention = self._encode_chunk(*encode_args)
            selected_chunks.append(selected)
            attention_chunks.append(attention)
            pair_mask_chunks.append(pair_valid)

        return {
            "selected": torch.cat(selected_chunks, dim=1),
            "attention": torch.cat(attention_chunks, dim=1),
            "pair_mask": torch.cat(pair_mask_chunks, dim=1),
            "indices_a": indices_a,
            "indices_b": indices_b,
            "valid_a": valid_a,
            "valid_b": valid_b,
            "selected_types_a": selected_types_a,
            "selected_types_b": selected_types_b,
            "source_types_a": types_a,
            "source_types_b": types_b,
            "source_mask_a": mask_a,
            "source_mask_b": mask_b,
        }

    def _forward_precomputed(
        self,
        query,
        indices_a,
        valid_a,
        indices_b,
        valid_b,
        types_a,
        mask_a,
        types_b,
        mask_b,
        pair_tokens,
        pair_mask,
        drug_b_count,
    ):
        selected_types_a = self._gather_candidate(
            types_a.unsqueeze(-1), indices_a
        ).squeeze(-1)
        selected_types_b = self._gather_candidate(
            types_b.unsqueeze(-1), indices_b
        ).squeeze(-1)
        selected_chunks = []
        attention_chunks = []
        pair_mask_chunks = []
        for start in range(0, query.size(1), self.candidate_chunk_size):
            end = min(start + self.candidate_chunk_size, query.size(1))
            linear_indices = (
                indices_a[:, start:end].unsqueeze(-1)
                * drug_b_count[:, None, None, None]
                + indices_b[:, start:end].unsqueeze(-2)
            ).reshape(query.size(0), end - start, -1)
            real_valid = (
                valid_a[:, start:end].unsqueeze(-1)
                & valid_b[:, start:end].unsqueeze(-2)
            ).reshape(query.size(0), end - start, -1)
            safe_indices = linear_indices.clamp_max(pair_tokens.size(1) - 1)
            selected_pair_mask = torch.gather(
                pair_mask.unsqueeze(1).expand(-1, end - start, -1),
                2,
                safe_indices,
            )
            real_valid = real_valid & selected_pair_mask
            aggregate_mask = real_valid
            if self.use_null_evidence:
                aggregate_mask = torch.cat(
                    [
                        aggregate_mask,
                        torch.ones(
                            query.size(0),
                            end - start,
                            1,
                            dtype=torch.bool,
                            device=query.device,
                        ),
                    ],
                    dim=-1,
                )
            aggregate_args = (
                query[:, start:end],
                pair_tokens,
                safe_indices,
                aggregate_mask,
            )
            if self.training:
                selected, attention = checkpoint(
                    self._aggregate_precomputed_chunk, *aggregate_args
                )
            else:
                selected, attention = self._aggregate_precomputed_chunk(
                    *aggregate_args
                )
            selected_chunks.append(selected)
            attention_chunks.append(attention)
            pair_mask_chunks.append(aggregate_mask)
        return {
            "selected": torch.cat(selected_chunks, dim=1),
            "attention": torch.cat(attention_chunks, dim=1),
            "pair_mask": torch.cat(pair_mask_chunks, dim=1),
            "indices_a": indices_a,
            "indices_b": indices_b,
            "valid_a": valid_a,
            "valid_b": valid_b,
            "selected_types_a": selected_types_a,
            "selected_types_b": selected_types_b,
            "source_types_a": types_a,
            "source_types_b": types_b,
            "source_mask_a": mask_a,
            "source_mask_b": mask_b,
        }


class CandidateMatchingHead(nn.Module):
    """Score each candidate DDIE with DDIE-specific selected drug evidence."""

    def __init__(self, pair_dim, event_dim, evidence_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.event_pool = nn.Identity()
        self.scorer = nn.Sequential(
            nn.Linear(pair_dim + event_dim + evidence_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pair_repr, event_tokens, selected_evidence):
        """
        Args:
            pair_repr: [batch, pair_dim]
            event_tokens: [num_events, event_len, event_dim] or [num_events, event_dim]
            selected_evidence: [batch, num_events, evidence_dim]

        Returns:
            logits: [batch, num_events]
        """
        event_repr = pool_event_tokens(event_tokens)

        batch_size = pair_repr.size(0)
        num_events = event_repr.size(0)

        pair_expand = pair_repr.unsqueeze(1).expand(batch_size, num_events, -1)
        event_expand = event_repr.unsqueeze(0).expand(batch_size, num_events, -1)
        features = torch.cat([pair_expand, event_expand, selected_evidence], dim=-1)
        return self.scorer(features).squeeze(-1)


class ReverseAttentionCandidateMatcher(nn.Module):
    """Minimal DDIE-query-to-drug-evidence matching model."""

    def __init__(
        self,
        pair_dim=256,
        evidence_dim=300,
        event_dim=256,
        hidden_dim=256,
        dropout=0.1,
        use_null_evidence=True,
        use_substructure_evidence=True,
        use_evidence_gate=False,
        use_kg_evidence=False,
        kg_evidence_dim=300,
        kg_hidden_dim=None,
        kg_feature_vocab_sizes=None,
        use_pharmacophore_evidence=False,
        pharmacophore_evidence_dim=300,
        pharmacophore_hidden_dim=None,
        use_pharmacophore_gate=False,
        pharmacophore_top_k=None,
        pharmacophore_drug_top_k=None,
        pharmacophore_type_dim=32,
        pharmacophore_candidate_chunk_size=8,
        pharmacophore_shared_pair_encoder=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_substructure_evidence = use_substructure_evidence
        self.use_evidence_gate = use_evidence_gate
        if self.use_evidence_gate and not self.use_substructure_evidence:
            raise ValueError(
                "use_evidence_gate requires use_substructure_evidence=True"
            )
        self.use_kg_evidence = use_kg_evidence
        self.kg_hidden_dim = kg_hidden_dim or hidden_dim
        self.use_pharmacophore_evidence = use_pharmacophore_evidence
        self.pharmacophore_hidden_dim = pharmacophore_hidden_dim or hidden_dim
        self.use_pharmacophore_gate = use_pharmacophore_gate
        self.pharmacophore_top_k = pharmacophore_top_k
        self.pharmacophore_drug_top_k = pharmacophore_drug_top_k
        if pharmacophore_top_k is not None and pharmacophore_drug_top_k is not None:
            raise ValueError(
                "pair-level and drug-level pharmacophore Top-K are mutually exclusive"
            )
        if self.use_pharmacophore_gate and not self.use_pharmacophore_evidence:
            raise ValueError(
                "use_pharmacophore_gate requires use_pharmacophore_evidence=True"
            )
        self.kg_feature_encoder = None
        if self.use_substructure_evidence:
            self.selector = DDIEGuidedEvidenceSelector(
                evidence_dim=evidence_dim,
                event_dim=event_dim,
                hidden_dim=hidden_dim,
                use_null_evidence=use_null_evidence,
            )
        if self.use_kg_evidence:
            if kg_feature_vocab_sizes is not None:
                self.kg_feature_encoder = KGEvidenceFeatureEncoder(
                    vocab_sizes=kg_feature_vocab_sizes,
                    output_dim=kg_evidence_dim,
                    dropout=dropout,
                )
            self.kg_selector = DDIEGuidedEvidenceSelector(
                evidence_dim=kg_evidence_dim,
                event_dim=event_dim,
                hidden_dim=self.kg_hidden_dim,
                use_null_evidence=use_null_evidence,
            )
        if self.use_pharmacophore_evidence:
            if self.pharmacophore_drug_top_k is not None:
                self.pharmacophore_drug_selector = CandidateSpecificDrugPairSelector(
                    atom_dim=pharmacophore_evidence_dim,
                    event_dim=event_dim,
                    pair_dim=pharmacophore_evidence_dim,
                    output_dim=self.pharmacophore_hidden_dim,
                    top_k=self.pharmacophore_drug_top_k,
                    type_dim=pharmacophore_type_dim,
                    dropout=dropout,
                    candidate_chunk_size=pharmacophore_candidate_chunk_size,
                    use_null_evidence=use_null_evidence,
                    shared_pair_encoder=pharmacophore_shared_pair_encoder,
                )
            else:
                self.pharmacophore_selector = DDIEGuidedEvidenceSelector(
                    evidence_dim=pharmacophore_evidence_dim,
                    event_dim=event_dim,
                    hidden_dim=self.pharmacophore_hidden_dim,
                    use_null_evidence=use_null_evidence,
                )
            if self.use_pharmacophore_gate:
                self.pharmacophore_gate = nn.Sequential(
                    nn.Linear(
                        pair_dim + event_dim + self.pharmacophore_hidden_dim,
                        hidden_dim,
                    ),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid(),
                )
        if self.use_evidence_gate:
            self.evidence_gate = nn.Sequential(
                nn.Linear(pair_dim + event_dim + hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        self.scorer = CandidateMatchingHead(
            pair_dim=pair_dim,
            event_dim=event_dim,
            evidence_dim=(
                (hidden_dim if self.use_substructure_evidence else 0)
                + (self.kg_hidden_dim if self.use_kg_evidence else 0)
                + (
                    self.pharmacophore_hidden_dim
                    if self.use_pharmacophore_evidence
                    else 0
                )
            ),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        pair_repr,
        evidence_tokens,
        event_tokens,
        labels=None,
        evidence_mask=None,
        kg_evidence_tokens=None,
        kg_evidence_mask=None,
        pharmacophore_evidence_tokens=None,
        pharmacophore_evidence_mask=None,
        pharmacophore_valid_pair_count=None,
        pharmacophore_drug_a_nodes=None,
        pharmacophore_drug_a_types=None,
        pharmacophore_drug_a_mask=None,
        pharmacophore_drug_b_nodes=None,
        pharmacophore_drug_b_types=None,
        pharmacophore_drug_b_mask=None,
        pharmacophore_drug_b_count=None,
    ):
        if self.use_substructure_evidence:
            if evidence_tokens is None:
                raise ValueError(
                    "evidence_tokens are required when substructure evidence is enabled"
                )
            selected, attention, _, _ = self.selector(
                event_tokens, evidence_tokens, evidence_mask
            )
        else:
            event_repr = pool_event_tokens(event_tokens)
            selected = pair_repr.new_zeros(
                pair_repr.size(0), event_repr.size(0), self.hidden_dim
            )
            attention = None
        kg_selected = None
        kg_attention = None
        if self.use_kg_evidence:
            if kg_evidence_tokens is None:
                event_repr = pool_event_tokens(event_tokens)
                kg_selected = selected.new_zeros(
                    selected.size(0),
                    event_repr.size(0),
                    self.kg_hidden_dim,
                )
            else:
                if self.kg_feature_encoder is not None:
                    kg_evidence_tokens = self.kg_feature_encoder(kg_evidence_tokens)
                kg_selected, kg_attention, _, _ = self.kg_selector(
                    event_tokens,
                    kg_evidence_tokens,
                    kg_evidence_mask,
                )
        pharmacophore_selected = None
        pharmacophore_attention = None
        pharmacophore_selection_indices = None
        pharmacophore_selection_mask = None
        pharmacophore_drug_selection = None
        pharmacophore_gate = None
        if self.use_pharmacophore_evidence:
            if self.pharmacophore_drug_top_k is not None:
                required = (
                    pharmacophore_drug_a_nodes,
                    pharmacophore_drug_a_types,
                    pharmacophore_drug_a_mask,
                    pharmacophore_drug_b_nodes,
                    pharmacophore_drug_b_types,
                    pharmacophore_drug_b_mask,
                    pharmacophore_drug_b_count,
                )
                if any(value is None for value in required):
                    raise ValueError(
                        "drug-level pharmacophore Top-K requires both drugs' nodes, types, and masks"
                    )
                pharmacophore_drug_selection = self.pharmacophore_drug_selector(
                    event_tokens,
                    pharmacophore_drug_a_nodes,
                    pharmacophore_drug_a_types,
                    pharmacophore_drug_a_mask,
                    pharmacophore_drug_b_nodes,
                    pharmacophore_drug_b_types,
                    pharmacophore_drug_b_mask,
                    precomputed_pair_tokens=pharmacophore_evidence_tokens,
                    precomputed_pair_mask=pharmacophore_evidence_mask,
                    drug_b_count=pharmacophore_drug_b_count,
                )
                pharmacophore_selected = pharmacophore_drug_selection["selected"]
                pharmacophore_attention = pharmacophore_drug_selection["attention"]
                pharmacophore_selection_mask = pharmacophore_drug_selection["pair_mask"]
            elif pharmacophore_evidence_tokens is None:
                event_repr = pool_event_tokens(event_tokens)
                pharmacophore_selected = selected.new_zeros(
                    selected.size(0),
                    event_repr.size(0),
                    self.pharmacophore_hidden_dim,
                )
            else:
                (
                    pharmacophore_selected,
                    pharmacophore_attention,
                    pharmacophore_selection_indices,
                    pharmacophore_selection_mask,
                ) = self.pharmacophore_selector(
                    event_tokens,
                    pharmacophore_evidence_tokens,
                    pharmacophore_evidence_mask,
                    top_k=self.pharmacophore_top_k,
                )
            if self.use_pharmacophore_gate:
                event_repr = pool_event_tokens(event_tokens)
                batch_size = pair_repr.size(0)
                num_events = event_repr.size(0)
                pair_expand = pair_repr.unsqueeze(1).expand(batch_size, num_events, -1)
                event_expand = event_repr.unsqueeze(0).expand(batch_size, num_events, -1)
                gate_input = torch.cat(
                    [pair_expand, event_expand, pharmacophore_selected],
                    dim=-1,
                )
                pharmacophore_gate = self.pharmacophore_gate(gate_input)
                pharmacophore_selected = pharmacophore_selected * pharmacophore_gate

        evidence_gate = None
        if self.use_evidence_gate:
            event_repr = pool_event_tokens(event_tokens)
            batch_size = pair_repr.size(0)
            num_events = event_repr.size(0)
            pair_expand = pair_repr.unsqueeze(1).expand(batch_size, num_events, -1)
            event_expand = event_repr.unsqueeze(0).expand(batch_size, num_events, -1)
            gate_input = torch.cat([pair_expand, event_expand, selected], dim=-1)
            evidence_gate = self.evidence_gate(gate_input)
            selected = selected * evidence_gate

        selected_for_score = selected if self.use_substructure_evidence else selected[..., :0]
        if self.use_kg_evidence:
            selected_for_score = torch.cat([selected_for_score, kg_selected], dim=-1)
        if self.use_pharmacophore_evidence:
            selected_for_score = torch.cat(
                [selected_for_score, pharmacophore_selected],
                dim=-1,
            )

        logits = self.scorer(pair_repr, event_tokens, selected_for_score)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "selected_evidence": selected,
            "attention": attention,
            "kg_selected_evidence": kg_selected,
            "kg_attention": kg_attention,
            "pharmacophore_selected_evidence": pharmacophore_selected,
            "pharmacophore_attention": pharmacophore_attention,
            "pharmacophore_selection_indices": pharmacophore_selection_indices,
            "pharmacophore_selection_mask": pharmacophore_selection_mask,
            "pharmacophore_drug_selection": pharmacophore_drug_selection,
            "pharmacophore_valid_pair_count": (
                pharmacophore_valid_pair_count
                if pharmacophore_valid_pair_count is not None
                else (
                    pharmacophore_evidence_mask.sum(dim=-1)
                    if pharmacophore_evidence_mask is not None
                    else None
                )
            ),
            "pharmacophore_gate": pharmacophore_gate,
            "evidence_gate": evidence_gate,
        }
