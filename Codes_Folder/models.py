"""Comprehensive PyTorch Implementations of All 10 Deep Learning & Machine Learning Algorithms.

Project: Predictive Maintenance in Metro Trains Using Machine Learning and Deep Learning
Author: Moorthy Chetan (Reg. No: 24BIT0246)

The 10 Algorithms:
 1. LSTM (Long Short-Term Memory)
 2. GRU (Gated Recurrent Unit)
 3. BiLSTM (Bidirectional LSTM)
 4. CNN-LSTM (1D Convolutional Neural Network + LSTM)
 5. CNN-BiLSTM with Attention (1D CNN + BiLSTM + Additive Attention Mechanism)
 6. Temporal Convolutional Network (TCN - Dilated Causal Residual Convolutions)
 7. Transformer Encoder (Multi-Head Self-Attention + Positional Encoding)
 8. Temporal Fusion Transformer (TFT - Gated Residual Networks + Variable Selection + Temporal Attention)
 9. LSTM Autoencoder (Sequence-to-Sequence Reconstruction-based Anomaly Detector)
10. Variational Autoencoder (VAE - Probabilistic Latent Generative Reconstruction Model)
"""

import math
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 1. Long Short-Term Memory (LSTM)
# -----------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    """2-layer Long Short-Term Memory network for multivariate sequence classification."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.name = "LSTM"
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len=60, input_dim=16]
        out, (hn, cn) = self.lstm(x)
        last_hidden = out[:, -1, :]  # [batch_size, hidden_dim]
        logits = self.fc(last_hidden).squeeze(-1)
        return logits


# -----------------------------------------------------------------------------
# 2. Gated Recurrent Unit (GRU)
# -----------------------------------------------------------------------------
class GRUClassifier(nn.Module):
    """2-layer Gated Recurrent Unit network for efficient sequence classification."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.name = "GRU"
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, hn = self.gru(x)
        last_hidden = out[:, -1, :]
        logits = self.fc(last_hidden).squeeze(-1)
        return logits


# -----------------------------------------------------------------------------
# 3. Bidirectional LSTM (BiLSTM)
# -----------------------------------------------------------------------------
class BiLSTMClassifier(nn.Module):
    """2-layer Bidirectional LSTM capturing both forward and reverse temporal context."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.name = "BiLSTM"
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Bidirectional concatenates forward and backward hidden states: hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (hn, cn) = self.lstm(x)
        # Concatenate final forward hidden state and first backward hidden state
        forward_feat = out[:, -1, :self.lstm.hidden_size]
        backward_feat = out[:, 0, self.lstm.hidden_size:]
        bi_hidden = torch.cat([forward_feat, backward_feat], dim=-1)
        logits = self.fc(bi_hidden).squeeze(-1)
        return logits


# -----------------------------------------------------------------------------
# 4. CNN-LSTM (1D Convolutional Neural Network + LSTM)
# -----------------------------------------------------------------------------
class CNNLSTMClassifier(nn.Module):
    """Hybrid CNN-LSTM: 1D Convolutions extract local sensor motifs followed by LSTM sequence modeling."""

    def __init__(self, input_dim: int = 16, cnn_channels: int = 32, lstm_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.name = "CNN-LSTM"
        # 1D Convolution over temporal dimension
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, input_dim] -> transpose to [B, input_dim, seq_len] for Conv1d
        x_trans = x.transpose(1, 2)
        conv_out = self.conv(x_trans)  # [B, cnn_channels, seq_len]
        # Transpose back for LSTM: [B, seq_len, cnn_channels]
        conv_trans = conv_out.transpose(1, 2)
        out, (hn, cn) = self.lstm(conv_trans)
        last_hidden = out[:, -1, :]
        logits = self.fc(last_hidden).squeeze(-1)
        return logits


# -----------------------------------------------------------------------------
# 5. CNN-BiLSTM with Attention
# -----------------------------------------------------------------------------
class AttentionLayer(nn.Module):
    """Additive Bahdanau-style self-attention mechanism over sequence timesteps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq_len, hidden_dim]
        scores = self.attn(x)  # [batch, seq_len, 1]
        weights = F.softmax(scores, dim=1)  # [batch, seq_len, 1]
        context = torch.sum(x * weights, dim=1)  # [batch, hidden_dim]
        return context, weights


class CNNBiLSTMAttentionClassifier(nn.Module):
    """1D CNN + Bidirectional LSTM with Temporal Attention Layer."""

    def __init__(self, input_dim: int = 16, cnn_channels: int = 32, lstm_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.name = "CNN-BiLSTM-Attention"
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attention = AttentionLayer(hidden_dim=lstm_hidden * 2)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv feature extraction
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        # BiLSTM representation
        bilstm_out, _ = self.bilstm(conv_out)  # [B, seq_len, lstm_hidden * 2]
        # Attention pooling
        context, _ = self.attention(bilstm_out)  # [B, lstm_hidden * 2]
        logits = self.fc(context).squeeze(-1)
        return logits


# -----------------------------------------------------------------------------
# 6. Temporal Convolutional Network (TCN)
# -----------------------------------------------------------------------------
class ChausalConv1d(nn.Module):
    """Dilated Causal 1D Convolution: pads strictly in the past to prevent lookahead."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Residual block for Temporal Convolutional Network."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = ChausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = ChausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        out = self.drop1(self.relu1(self.conv1(x)))
        out = self.drop2(self.relu2(self.conv2(out)))
        return self.final_relu(out + res)


class TCNClassifier(nn.Module):
    """Temporal Convolutional Network with dilated causal residual blocks."""

    def __init__(self, input_dim: int = 16, num_channels: list = [32, 64, 128], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.name = "TCN"
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=dilation_size, dropout=dropout))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_trans = x.transpose(1, 2)
        feat = self.network(x_trans)
        pooled = feat.mean(dim=-1)
        logits = self.fc(pooled).squeeze(-1)
        return logits


# -----------------------------------------------------------------------------
# 7. Transformer Encoder
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence order awareness."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderClassifier(nn.Module):
    """Multi-Head Self-Attention Transformer Encoder for multivariate time-series."""

    def __init__(self, input_dim: int = 16, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.name = "Transformer-Encoder"
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len=60, input_dim=16]
        proj = self.input_proj(x)
        encoded = self.pos_encoder(proj)
        trans_out = self.transformer_encoder(encoded)
        # Global Average Pooling across time
        pooled = trans_out.mean(dim=1)
        logits = self.fc(pooled).squeeze(-1)
        return logits


class CNNTransformerClassifier(nn.Module):
    """Hybrid CNN-Transformer: 1D CNN extracts local sensor motifs, output is stored in a variable 
    and fed into next sequence/stage Transformers."""

    def __init__(self, input_dim: int = 16, cnn_channels: int = 64, d_model: int = 64, nhead: int = 4, dropout: float = 0.2):
        super().__init__()
        self.name = "CNN-Transformer"
        
        # 1. 1D CNN Feature Extractor
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional Encoding for temporal order
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. Sequence Transformer Stage 1
        encoder_layer1 = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True
        )
        self.transformer_stage1 = nn.TransformerEncoder(encoder_layer1, num_layers=1)
        
        # 3. Next Sequence Transformer Stage 2
        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True
        )
        self.transformer_stage2 = nn.TransformerEncoder(encoder_layer2, num_layers=1)
        
        # 4. Classification Head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: 1D CNN extracts local sensor feature representations
        # Transpose to [B, channels, seq_len] for Conv1d, then back to [B, seq_len, channels]
        cnn_output = self.conv(x.transpose(1, 2)).transpose(1, 2)
        
        # Step 2: Store CNN output in variable and pass with positional encoding into Transformer Stage 1
        stage1_input = self.pos_encoder(cnn_output)
        transformer_stage1_output = self.transformer_stage1(stage1_input)
        
        # Step 3: Pass stored intermediate output into next sequence Transformer Stage 2
        transformer_stage2_output = self.transformer_stage2(transformer_stage1_output)
        
        # Step 4: Sequence temporal pooling & final classification logit
        pooled_features = transformer_stage2_output.mean(dim=1)
        logits = self.fc(pooled_features).squeeze(-1)
        return logits


# -----------------------------------------------------------------------------
# 8. Temporal Fusion Transformer (TFT)
# -----------------------------------------------------------------------------
class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) module used in Temporal Fusion Transformers."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        # Gated Linear Unit
        gated = torch.sigmoid(self.gate(h)) * h
        return self.layer_norm(skip + gated)


class TemporalFusionTransformerClassifier(nn.Module):
    """Temporal Fusion Transformer (TFT) with Gated Residual Networks, Variable Selection, and Self-Attention."""

    def __init__(self, input_dim: int = 16, d_model: int = 64, nhead: int = 4, dropout: float = 0.2):
        super().__init__()
        self.name = "Temporal-Fusion-Transformer"
        self.d_model = d_model
        # Feature transformation
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(1, d_model, d_model, dropout=dropout)
            for _ in range(input_dim)
        ])
        # Variable Selection Weighting
        self.var_weights_net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, input_dim)
        )
        # Temporal Self-Attention
        self.pos_encoder = PositionalEncoding(d_model)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=dropout)
        self.post_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len=60, input_dim=16]
        B, T, C = x.shape
        # Variable selection across the 16 features
        var_weights = F.softmax(self.var_weights_net(x), dim=-1)  # [B, T, C]
        
        # Transform each feature individually through GRN
        transformed = []
        for i in range(C):
            feat_i = x[:, :, i:i+1]  # [B, T, 1]
            t_i = self.var_grns[i](feat_i)  # [B, T, d_model]
            transformed.append(t_i.unsqueeze(-2))
        
        all_trans = torch.cat(transformed, dim=-2)  # [B, T, C, d_model]
        # Weighted combination of selected variables
        selected_vars = torch.sum(all_trans * var_weights.unsqueeze(-1), dim=-2)  # [B, T, d_model]
        
        # Temporal self-attention
        encoded = self.pos_encoder(selected_vars)
        attn_out, _ = self.temporal_attn(encoded, encoded, encoded)
        fused = self.post_grn(attn_out + encoded)
        
        # Pool across timesteps
        pooled = fused.mean(dim=1)
        logits = self.fc(pooled).squeeze(-1)
        return logits


# -----------------------------------------------------------------------------
# 9. LSTM Autoencoder (Reconstruction Anomaly Detector)
# -----------------------------------------------------------------------------
class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for reconstruction-based anomaly detection with classification head."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, latent_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.name = "LSTM-Autoencoder"
        self.latent_dim = latent_dim
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.recon_proj = nn.Linear(hidden_dim, input_dim)
        
        # Classification Head (trained simultaneously or on latent code + reconstruction error)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 1, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out, (hn, cn) = self.encoder_lstm(x)
        z = self.encoder_fc(out[:, -1, :])  # [B, latent_dim]
        return z

    def decode(self, z: torch.Tensor, seq_len: int = 60) -> torch.Tensor:
        h = self.decoder_fc(z).unsqueeze(1).repeat(1, seq_len, 1)  # [B, seq_len, hidden_dim]
        out, _ = self.decoder_lstm(h)
        recon = self.recon_proj(out)  # [B, seq_len, input_dim]
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len=60, input_dim=16]
        z = self.encode(x)
        recon = self.decode(z, seq_len=x.size(1))
        # Reconstruction MSE per sequence
        mse_error = torch.mean((x - recon) ** 2, dim=[1, 2]).unsqueeze(-1)  # [B, 1]
        # Combined latent code and reconstruction error for classification
        combined = torch.cat([z, mse_error], dim=-1)
        logits = self.fc(combined).squeeze(-1)
        return logits

    def compute_reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        recon = self.decode(z, seq_len=x.size(1))
        return F.mse_loss(recon, x)


# -----------------------------------------------------------------------------
# 10. Variational Autoencoder (VAE)
# -----------------------------------------------------------------------------
class VariationalAutoencoder(nn.Module):
    """Probabilistic Variational Autoencoder with reparameterization trick and ELBO optimization."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, latent_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.name = "Variational-Autoencoder"
        self.latent_dim = latent_dim
        
        # Encoder GRU
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder GRU
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.recon_proj = nn.Linear(hidden_dim, input_dim)
        
        # Classification Head
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 1, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.encoder(x)
        h_last = out[:, -1, :]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, seq_len: int = 60) -> torch.Tensor:
        h = self.decoder_input(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder(h)
        recon = self.recon_proj(out)
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, seq_len=x.size(1))
        # Reconstruction error
        recon_error = torch.mean((x - recon) ** 2, dim=[1, 2]).unsqueeze(-1)
        combined = torch.cat([z, recon_error], dim=-1)
        logits = self.fc(combined).squeeze(-1)
        return logits

    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, seq_len=x.size(1))
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        total_loss = recon_loss + 0.001 * kl_div
        return total_loss, recon_loss, kl_div


# -----------------------------------------------------------------------------
# Model Registry Mapping
# -----------------------------------------------------------------------------
MODEL_REGISTRY = {
    "lstm": LSTMClassifier,
    "gru": GRUClassifier,
    "bilstm": BiLSTMClassifier,
    "cnn_lstm": CNNLSTMClassifier,
    "cnn_bilstm_attn": CNNBiLSTMAttentionClassifier,
    "cnn_transformer": CNNTransformerClassifier,
    "tcn": TCNClassifier,
    "transformer": TransformerEncoderClassifier,
    "tft": TemporalFusionTransformerClassifier,
    "lstm_ae": LSTMAutoencoder,
    "vae": VariationalAutoencoder
}

ALGORITHM_DISPLAY_NAMES = {
    "lstm": "1. LSTM (Long Short-Term Memory)",
    "gru": "2. GRU (Gated Recurrent Unit)",
    "bilstm": "3. BiLSTM (Bidirectional LSTM)",
    "cnn_lstm": "4. CNN-LSTM (1D Convolution + LSTM)",
    "cnn_bilstm_attn": "5. CNN-BiLSTM with Attention",
    "cnn_transformer": "CNN-Transformer (CNN -> Var -> Next Seq Transformers)",
    "tcn": "6. Temporal Convolutional Network (TCN)",
    "transformer": "7. Transformer Encoder",
    "tft": "8. Temporal Fusion Transformer (TFT)",
    "lstm_ae": "9. LSTM Autoencoder",
    "vae": "10. Variational Autoencoder (VAE)"
}
