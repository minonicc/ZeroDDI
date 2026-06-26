import math

import torch
from torch import nn
import torch.nn.functional as F


class DDIEGuidedEvidenceSelector(nn.Module):
    """Select drug evidence tokens for each candidate DDIE representation."""

    def __init__(self, evidence_dim, event_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(event_dim, hidden_dim)
        self.key = nn.Linear(evidence_dim, hidden_dim)
        self.value = nn.Linear(evidence_dim, hidden_dim)

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
        if event_tokens.dim() == 3:
            event_repr = event_tokens.mean(dim=1)
        elif event_tokens.dim() == 2:
            event_repr = event_tokens
        else:
            raise ValueError("event_tokens must be [C, L, D] or [C, D]")

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
        if event_tokens.dim() == 3:
            event_repr = event_tokens.mean(dim=1)
        elif event_tokens.dim() == 2:
            event_repr = event_tokens
        else:
            raise ValueError("event_tokens must be [C, L, D] or [C, D]")

        batch_size = pair_repr.size(0)
        num_events = event_repr.size(0)

        pair_expand = pair_repr.unsqueeze(1).expand(batch_size, num_events, -1)
        event_expand = event_repr.unsqueeze(0).expand(batch_size, num_events, -1)
        features = torch.cat([pair_expand, event_expand, selected_evidence], dim=-1)
        return self.scorer(features).squeeze(-1)


class ReverseAttentionCandidateMatcher(nn.Module):
    """Minimal DDIE-query-to-drug-evidence matching model."""

    def __init__(self, pair_dim=256, evidence_dim=300, event_dim=256, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.selector = DDIEGuidedEvidenceSelector(
            evidence_dim=evidence_dim,
            event_dim=event_dim,
            hidden_dim=hidden_dim,
        )
        self.scorer = CandidateMatchingHead(
            pair_dim=pair_dim,
            event_dim=event_dim,
            evidence_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, pair_repr, evidence_tokens, event_tokens, labels=None, evidence_mask=None):
        selected, attention = self.selector(event_tokens, evidence_tokens, evidence_mask)
        logits = self.scorer(pair_repr, event_tokens, selected)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "selected_evidence": selected,
            "attention": attention,
        }
