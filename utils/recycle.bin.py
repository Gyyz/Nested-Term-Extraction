# This is a temp code recycle in case that the code can be used again

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hiddenDim = hidden_dim
        self.attSeq = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1))

    def forward(self, hidden_states):
        # (B, L, H) -> (B , L, 1)
        energy = self.attSeq(hidden_states)
        weights = F.softmax(energy.squeeze(-1), dim=0)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (hidden_states * weights.unsqueeze(-1)).sum(dim=0)
        return outputs, weights


class TermNumAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TermNumAttention, self).__init__()
        self.hiddenDim = hidden_dim
        self.termNumWeight = nn.Parameter(torch.Tensor(np.random.randn(hidden_dim)))
        self.attSeq = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(True), nn.Linear(64, 1))

    def forward(self, hidden_states):
        # after this, we have (batch, dim1) with a diff weight per each cell
        if hidden_states.size(0) == 0:
            return torch.zeros(self.hiddenDim)
        attention_hidden = hidden_states * self.termNumWeight
        attention_score = self.attSeq(attention_hidden)
        attention_score = F.softmax(attention_score, dim=1) # sigmoid(attention_score)
        scored_x = hidden_states * attention_score
        condensed_x = torch.sum(scored_x, dim=1)
        return condensed_x