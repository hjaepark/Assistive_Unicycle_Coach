# train/unicycle_train/coach/v5/autoencoder_model.py

"""
V5 Trajectory Autoencoder

Architecture (see chat history for design rationale):
  - Encoder: Bidirectional GRU → bottleneck linear → latent z
  - Decoder: z broadcast to all timesteps + learnable positional encoding
             → unidirectional GRU → linear → reconstructed trajectory
  - Loss:    masked MSE (padding positions excluded)

The encoder reads a full episode (state+action sequence) and compresses it
into a single latent z. The decoder reconstructs the full episode from z
alone. Training is fully self-supervised: no reward signal, no labels.

After training, the encoder is frozen and used to provide z to the V5
coach (π_I), augmenting its observation from [avg_reward, success_rate]
(V4) to [avg_reward, success_rate, z_1, ..., z_32] (V5).
"""

import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    """Bidirectional GRU encoder that compresses an episode into latent z.
    
    Input:  (B, T_max, feature_dim) padded trajectory
            (B, T_max)              mask, True at valid timesteps
    Output: (B, latent_dim)         single latent vector per episode
    """
    def __init__(
        self,
        feature_dim: int = 21,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Project per-timestep features into GRU input space
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Bidirectional GRU processes the sequence in both directions
        # output_size per direction = hidden_dim, so concat = 2*hidden_dim
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Bottleneck to latent dim
        # Input: 2 * hidden_dim (forward final + backward final)
        self.to_latent = nn.Linear(2 * hidden_dim, latent_dim)
    
    def forward(self, x, mask):
        """
        Args:
            x:    (B, T_max, feature_dim) float32, zero-padded
            mask: (B, T_max) bool, True for valid timesteps
        Returns:
            z: (B, latent_dim) float32
        """
        B, T_max, _ = x.shape
        
        # Project features
        h = self.input_proj(x)  # (B, T_max, hidden_dim)
        
        # Compute actual lengths from mask for packed sequence
        # (avoids GRU processing padding zeros, which would corrupt hidden state)
        lengths = mask.sum(dim=1).long()  # (B,)
        lengths_cpu = lengths.cpu()  # pack_padded_sequence wants CPU tensor
        
        # Pack sequence: GRU will skip padding
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        
        # GRU forward
        # h_n shape: (num_layers * 2, B, hidden_dim)
        # Last dim 2 = bidirectional (forward, backward)
        _, h_n = self.gru(packed)
        
        # We want the last layer's final hidden states (both directions)
        # h_n is organized as (layer_0_forward, layer_0_backward,
        #                     layer_1_forward, layer_1_backward, ...)
        # Last layer's forward = h_n[-2], backward = h_n[-1]
        forward_last = h_n[-2]   # (B, hidden_dim)
        backward_last = h_n[-1]  # (B, hidden_dim)
        
        # Concat and project to latent
        combined = torch.cat([forward_last, backward_last], dim=-1)  # (B, 2*hidden_dim)
        z = self.to_latent(combined)  # (B, latent_dim)
        
        return z


class TrajectoryDecoder(nn.Module):
    """Decoder: reconstruct full trajectory from latent z.
    
    z is broadcast to every timestep and combined with a learnable
    positional embedding so the decoder GRU can produce time-varying output
    despite receiving the same z at every step.
    
    Input:  z, shape (B, latent_dim)
    Output: (B, T_max, feature_dim) reconstructed trajectory
    """
    def __init__(
        self,
        feature_dim: int = 21,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        max_seq_len: int = 1500,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        
        # Project z to GRU input space
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Learnable positional embedding (one per timestep position)
        # Allows the decoder to distinguish t=10 from t=500 even though
        # both receive the same broadcasted z
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # GRU processes the broadcast + positional sequence
        # Unidirectional: at inference time the decoder runs forward only
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Project GRU output to reconstructed feature
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
    
    def forward(self, z, T_max=None):
        """
        Args:
            z:     (B, latent_dim)
            T_max: int, length of output sequence (defaults to self.max_seq_len)
        Returns:
            x_recon: (B, T_max, feature_dim)
        """
        B = z.shape[0]
        if T_max is None:
            T_max = self.max_seq_len
        
        # Project z to hidden_dim and broadcast to all timesteps
        z_h = self.z_proj(z)  # (B, hidden_dim)
        z_broadcast = z_h.unsqueeze(1).expand(-1, T_max, -1)  # (B, T_max, hidden_dim)
        
        # Positional embeddings for [0, 1, ..., T_max - 1]
        pos_ids = torch.arange(T_max, device=z.device)  # (T_max,)
        pos_emb = self.pos_embedding(pos_ids)  # (T_max, hidden_dim)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, T_max, hidden_dim)
        
        # Combine z broadcast with positional info
        gru_input = z_broadcast + pos_emb  # (B, T_max, hidden_dim)
        
        # GRU forward (no packing — decoder runs full length for parallel speed,
        # padded positions will be excluded from loss via mask later)
        gru_out, _ = self.gru(gru_input)  # (B, T_max, hidden_dim)
        
        # Project to feature space
        x_recon = self.output_proj(gru_out)  # (B, T_max, feature_dim)
        
        return x_recon


class TrajectoryAutoencoder(nn.Module):
    """Full autoencoder: encoder + decoder.
    
    Used during training. After training, only `encode()` is used by the
    V5 coach pipeline.
    """
    def __init__(
        self,
        feature_dim: int = 21,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        max_seq_len: int = 1500,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        
        self.encoder = TrajectoryEncoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.decoder = TrajectoryDecoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
    
    def encode(self, x, mask):
        return self.encoder(x, mask)
    
    def decode(self, z, T_max=None):
        return self.decoder(z, T_max=T_max)
    
    def forward(self, x, mask):
        """End-to-end forward: encode then decode.
        
        Args:
            x:    (B, T_max, feature_dim) padded trajectory
            mask: (B, T_max) bool mask
        Returns:
            x_recon: (B, T_max, feature_dim) reconstructed
            z:       (B, latent_dim) latent
        """
        z = self.encode(x, mask)
        x_recon = self.decode(z, T_max=x.shape[1])
        return x_recon, z


def masked_mse_loss(x_recon, x_target, mask):
    """Masked MSE loss: average over (valid timesteps × feature dims).
    
    Padding positions are excluded so the model is not penalized for
    reconstructing zeros that don't represent real trajectory data.
    
    Args:
        x_recon:  (B, T_max, feature_dim)
        x_target: (B, T_max, feature_dim)
        mask:     (B, T_max) bool, True at valid timesteps
    Returns:
        scalar loss
    """
    # Per-element squared error: (B, T_max, feature_dim)
    sq_err = (x_recon - x_target) ** 2
    
    # Broadcast mask to feature dim and convert to float
    mask_f = mask.unsqueeze(-1).float()  # (B, T_max, 1)
    
    # Sum over masked positions, divide by total number of masked elements
    # (multiplied by feature_dim so we get a per-element average)
    n_valid_elements = mask_f.sum() * x_recon.shape[-1]
    loss = (sq_err * mask_f).sum() / n_valid_elements.clamp(min=1.0)
    
    return loss


def count_parameters(model):
    """Count trainable parameters. Useful for sanity check."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test: build model, run dummy forward, verify shapes
    model = TrajectoryAutoencoder(
        feature_dim=21,
        hidden_dim=128,
        latent_dim=32,
        num_layers=2,
        max_seq_len=1500,
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"  Encoder: {count_parameters(model.encoder):,}")
    print(f"  Decoder: {count_parameters(model.decoder):,}")
    
    # Dummy batch: 4 episodes of varying lengths
    B, T_max, D = 4, 1500, 21
    x = torch.randn(B, T_max, D)
    
    # Variable-length mask: episodes of length 300, 500, 800, 1200
    mask = torch.zeros(B, T_max, dtype=torch.bool)
    lengths = [300, 500, 800, 1200]
    for i, L in enumerate(lengths):
        mask[i, :L] = True
    
    print(f"\nDummy forward pass:")
    print(f"  Input x:    {x.shape}")
    print(f"  Input mask: {mask.shape}, lengths = {lengths}")
    
    x_recon, z = model(x, mask)
    print(f"  Latent z:   {z.shape}")
    print(f"  Recon x:    {x_recon.shape}")
    
    loss = masked_mse_loss(x_recon, x, mask)
    print(f"  Loss:       {loss.item():.4f}")
    
    print(f"\n[OK] Forward pass successful")