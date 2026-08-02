import math

import torch
from torch import nn
import torch.nn.functional as F


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

    def forward(self, event_tokens, evidence_tokens, evidence_mask=None):
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
        return selected, attention


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
        use_evidence_gate=False,
        use_kg_evidence=False,
        kg_evidence_dim=300,
        kg_hidden_dim=None,
        kg_feature_vocab_sizes=None,
        use_pharmacophore_evidence=False,
        pharmacophore_evidence_dim=300,
        pharmacophore_hidden_dim=None,
        use_pharmacophore_gate=False,
    ):
        super().__init__()
        self.use_evidence_gate = use_evidence_gate
        self.use_kg_evidence = use_kg_evidence
        self.kg_hidden_dim = kg_hidden_dim or hidden_dim
        self.use_pharmacophore_evidence = use_pharmacophore_evidence
        self.pharmacophore_hidden_dim = pharmacophore_hidden_dim or hidden_dim
        self.use_pharmacophore_gate = use_pharmacophore_gate
        if self.use_pharmacophore_gate and not self.use_pharmacophore_evidence:
            raise ValueError(
                "use_pharmacophore_gate requires use_pharmacophore_evidence=True"
            )
        self.kg_feature_encoder = None
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
                hidden_dim
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
    ):
        selected, attention = self.selector(event_tokens, evidence_tokens, evidence_mask)
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
                kg_selected, kg_attention = self.kg_selector(
                    event_tokens,
                    kg_evidence_tokens,
                    kg_evidence_mask,
                )
        pharmacophore_selected = None
        pharmacophore_attention = None
        pharmacophore_gate = None
        if self.use_pharmacophore_evidence:
            if pharmacophore_evidence_tokens is None:
                event_repr = pool_event_tokens(event_tokens)
                pharmacophore_selected = selected.new_zeros(
                    selected.size(0),
                    event_repr.size(0),
                    self.pharmacophore_hidden_dim,
                )
            else:
                pharmacophore_selected, pharmacophore_attention = self.pharmacophore_selector(
                    event_tokens,
                    pharmacophore_evidence_tokens,
                    pharmacophore_evidence_mask,
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

        selected_for_score = selected
        if self.use_kg_evidence:
            selected_for_score = torch.cat([selected, kg_selected], dim=-1)
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
            "pharmacophore_gate": pharmacophore_gate,
            "evidence_gate": evidence_gate,
        }
