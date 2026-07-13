"""
ViTacDreamer: Conditional Variational Autoencoder for Robust Visual-Tactile Manipulation

This module implements the core cVAE architecture that uses tactile and action history
to reconstruct visual observations under distraction.

Legacy note:
- this is the original v1 implementation
- active temporal multitask work should target `vitacdreamer/model_v2.py`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from transformers import ViTModel, ViTConfig
import math
import os
from pathlib import Path


class ActionTokenizer(nn.Module):
    """Tokenizes action sequences using learned embeddings."""

    def __init__(
        self,
        action_dim: int = 7,
        embed_dim: int = 768,
        max_seq_len: int = 10,
        use_fast_tokenizer: bool = False
    ):
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_fast_tokenizer = use_fast_tokenizer

        if use_fast_tokenizer:
            # TODO: Integrate FAST tokenizer from physical-intelligence
            # For now, use learned embedding
            print("Warning: FAST tokenizer not yet integrated, using learned embedding")

        # Linear projection for actions
        self.action_proj = nn.Linear(action_dim, embed_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: (B, T, action_dim) action sequence

        Returns:
            (B, T, embed_dim) tokenized actions
        """
        B, T, _ = actions.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        # Project actions
        tokens = self.action_proj(actions)

        # Add positional encoding
        tokens = tokens + self.pos_encoding[:, :T, :]

        # Normalize
        tokens = self.norm(tokens)

        return tokens


class ModalityEncoder(nn.Module):
    """Encodes visual or tactile observations using ViT backbone."""

    @staticmethod
    def _resolve_vit_checkpoint() -> str:
        env_checkpoint = os.environ.get("VITACDREAMER_VIT_CHECKPOINT")
        if env_checkpoint:
            env_path = Path(env_checkpoint)
            if env_path.exists():
                return str(env_path)

        repo_local = Path(__file__).resolve().parents[1] / ".hf_cache" / "google_vit_base_patch16_224"
        if repo_local.exists():
            return str(repo_local)

        return "google/vit-base-patch16-224"

    def __init__(
        self,
        modality: str = 'visual',
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        pretrained: bool = True
    ):
        super().__init__()
        self.modality = modality

        if pretrained:
            # Use pretrained ViT
            vit_checkpoint = self._resolve_vit_checkpoint()
            local_only = Path(vit_checkpoint).exists()
            self.vit = ViTModel.from_pretrained(vit_checkpoint, local_files_only=local_only)
            # Adjust input channels if needed
            if num_channels != 3:
                old_conv = self.vit.embeddings.patch_embeddings.projection
                self.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
                    num_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride
                )
        else:
            # Create ViT from scratch
            config = ViTConfig(
                image_size=image_size,
                patch_size=patch_size,
                num_channels=num_channels,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads
            )
            self.vit = ViTModel(config)

        self.hidden_size = self.vit.config.hidden_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) images

        Returns:
            (B, hidden_size) encoded features (CLS token)
        """
        outputs = self.vit(pixel_values=images)
        # Use CLS token
        cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token

    def get_all_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Get all patch tokens (not just CLS)."""
        outputs = self.vit(pixel_values=images)
        return outputs.last_hidden_state


class ViTacDreamerEncoder(nn.Module):
    """
    Encoder for ViTacDreamer that processes:
    - Current tactile observation
    - Action history
    - Visual observation history
    """

    def __init__(
        self,
        visual_encoder: ModalityEncoder,
        tactile_encoder: ModalityEncoder,
        action_tokenizer: ActionTokenizer,
        latent_dim: int = 256,
        history_len: int = 5,
        use_transformer: bool = True,
        num_transformer_layers: int = 4
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.tactile_encoder = tactile_encoder
        self.action_tokenizer = action_tokenizer
        self.latent_dim = latent_dim
        self.history_len = history_len

        embed_dim = visual_encoder.hidden_size

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Modality embeddings
        self.visual_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.tactile_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.action_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.target_visual_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Temporal embeddings
        self.temporal_embed = nn.Parameter(torch.zeros(1, history_len + 1, embed_dim))

        if use_transformer:
            # Transformer for cross-modal fusion
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        else:
            self.transformer = None

        # Project to latent space (mean and logvar for VAE)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

    def forward(
        self,
        current_tactile: torch.Tensor,
        visual_history: torch.Tensor,
        tactile_history: torch.Tensor,
        action_history: torch.Tensor,
        current_visual: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            current_tactile: (B, C, H, W) current tactile observation
            visual_history: (B, T, C, H, W) visual observation history
            tactile_history: (B, T, C, H, W) tactile observation history
            action_history: (B, T, action_dim) action history
            current_visual: (B, C, H, W) current visual (for training only)

        Returns:
            mu: (B, latent_dim) mean of latent distribution
            logvar: (B, latent_dim) log variance of latent distribution
        """
        B = current_tactile.shape[0]
        T = visual_history.shape[1]

        # Encode current tactile
        current_tactile_feat = self.tactile_encoder(current_tactile)  # (B, embed_dim)
        current_tactile_feat = current_tactile_feat.unsqueeze(1)  # (B, 1, embed_dim)

        # Encode visual history
        visual_history_flat = visual_history.reshape(B * T, *visual_history.shape[2:])
        visual_history_feat = self.visual_encoder(visual_history_flat)  # (B*T, embed_dim)
        visual_history_feat = visual_history_feat.reshape(B, T, -1)  # (B, T, embed_dim)

        # Encode tactile history
        tactile_history_flat = tactile_history.reshape(B * T, *tactile_history.shape[2:])
        tactile_history_feat = self.tactile_encoder(tactile_history_flat)  # (B*T, embed_dim)
        tactile_history_feat = tactile_history_feat.reshape(B, T, -1)  # (B, T, embed_dim)

        # Tokenize actions
        action_tokens = self.action_tokenizer(action_history)  # (B, T, embed_dim)

        # Add modality embeddings
        current_tactile_feat = current_tactile_feat + self.tactile_embed
        visual_history_feat = visual_history_feat + self.visual_embed
        tactile_history_feat = tactile_history_feat + self.tactile_embed
        action_tokens = action_tokens + self.action_embed

        # Combine all tokens
        # Interleave visual, tactile, and action for each timestep
        tokens = []
        for t in range(T):
            tokens.append(visual_history_feat[:, t:t+1, :])
            tokens.append(tactile_history_feat[:, t:t+1, :])
            tokens.append(action_tokens[:, t:t+1, :])

        # Add current tactile
        tokens.append(current_tactile_feat)

        # During cVAE training, condition the posterior on the visual target.
        if current_visual is not None:
            current_visual_feat = self.visual_encoder(current_visual).unsqueeze(1)
            current_visual_feat = current_visual_feat + self.target_visual_embed
            tokens.append(current_visual_feat)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens.insert(0, cls_tokens)

        # Concatenate all tokens
        all_tokens = torch.cat(tokens, dim=1)  # (B, 1 + 3*T + 1, embed_dim)

        # Apply transformer
        if self.transformer is not None:
            all_tokens = self.transformer(all_tokens)

        # Use CLS token for latent encoding
        cls_output = all_tokens[:, 0, :]  # (B, embed_dim)

        # Project to latent space
        mu = self.fc_mu(cls_output)
        logvar = self.fc_logvar(cls_output)

        return mu, logvar


class ViTacDreamerDecoder(nn.Module):
    """
    Decoder for ViTacDreamer that reconstructs visual observation
    from latent code conditioned on tactile and action.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        embed_dim: int = 768,
        output_size: Tuple[int, int] = (224, 224),
        output_channels: int = 3,
        num_transformer_layers: int = 6,
        patch_size: int = 16
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.output_size = output_size
        self.output_channels = output_channels
        self.patch_size = patch_size

        # Calculate number of patches
        self.num_patches_h = output_size[0] // patch_size
        self.num_patches_w = output_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Project latent to embedding dimension
        self.latent_proj = nn.Linear(latent_dim, embed_dim)

        # Learnable patch queries
        self.patch_queries = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Positional encoding for patches
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_transformer_layers)

        # Project patches to pixels
        self.patch_proj = nn.Linear(embed_dim, patch_size * patch_size * output_channels)

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) latent code
            condition: (B, embed_dim) conditioning information

        Returns:
            (B, C, H, W) reconstructed visual observation
        """
        B = z.shape[0]

        # Project latent
        z_proj = self.latent_proj(z).unsqueeze(1)  # (B, 1, embed_dim)

        # Combine latent and condition as memory
        memory = torch.cat([z_proj, condition.unsqueeze(1)], dim=1)  # (B, 2, embed_dim)

        # Prepare patch queries
        queries = self.patch_queries.expand(B, -1, -1) + self.pos_encoding  # (B, num_patches, embed_dim)

        # Decode
        patch_features = self.transformer_decoder(queries, memory)  # (B, num_patches, embed_dim)

        # Project to pixels
        patches = self.patch_proj(patch_features)  # (B, num_patches, patch_size^2 * C)

        # Reshape to image
        patches = patches.reshape(B, self.num_patches_h, self.num_patches_w,
                                   self.patch_size, self.patch_size, self.output_channels)
        patches = patches.permute(0, 5, 1, 3, 2, 4)  # (B, C, H_p, patch_size, W_p, patch_size)
        image = patches.reshape(B, self.output_channels, self.output_size[0], self.output_size[1])

        return torch.sigmoid(image)


class ViTacDreamer(nn.Module):
    """
    Complete ViTacDreamer model: cVAE for visual reconstruction
    from tactile and action history.
    """

    def __init__(
        self,
        visual_image_size: int = 224,
        tactile_image_size: int = 224,
        action_dim: int = 7,
        latent_dim: int = 256,
        history_len: int = 5,
        pretrained_encoders: bool = True
    ):
        super().__init__()

        # Create encoders
        self.visual_encoder = ModalityEncoder(
            modality='visual',
            image_size=visual_image_size,
            pretrained=pretrained_encoders
        )

        self.tactile_encoder = ModalityEncoder(
            modality='tactile',
            image_size=tactile_image_size,
            pretrained=pretrained_encoders
        )

        # Action tokenizer
        self.action_tokenizer = ActionTokenizer(
            action_dim=action_dim,
            embed_dim=self.visual_encoder.hidden_size,
            max_seq_len=history_len,
        )

        # Main encoder
        self.encoder = ViTacDreamerEncoder(
            visual_encoder=self.visual_encoder,
            tactile_encoder=self.tactile_encoder,
            action_tokenizer=self.action_tokenizer,
            latent_dim=latent_dim,
            history_len=history_len
        )

        # Decoder
        self.decoder = ViTacDreamerDecoder(
            latent_dim=latent_dim,
            embed_dim=self.visual_encoder.hidden_size,
            output_size=(visual_image_size, visual_image_size)
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        current_tactile: torch.Tensor,
        visual_history: torch.Tensor,
        tactile_history: torch.Tensor,
        action_history: torch.Tensor,
        current_visual: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns dict with:
            - recon: reconstructed visual observation
            - mu: latent mean
            - logvar: latent log variance
            - z: sampled latent code
        """
        # Conditional prior p(z | tactile, history, action)
        prior_mu, prior_logvar = self.encoder(
            current_tactile=current_tactile,
            visual_history=visual_history,
            tactile_history=tactile_history,
            action_history=action_history,
            current_visual=None
        )

        # Posterior q(z | current_visual, tactile, history, action) during training.
        if current_visual is not None:
            mu, logvar = self.encoder(
                current_tactile=current_tactile,
                visual_history=visual_history,
                tactile_history=tactile_history,
                action_history=action_history,
                current_visual=current_visual
            )
        else:
            mu, logvar = prior_mu, prior_logvar

        z = self.reparameterize(mu, logvar)

        # Get conditioning from current tactile
        with torch.no_grad():
            condition = self.tactile_encoder(current_tactile)

        # Decode
        recon = self.decoder(z, condition)

        return {
            'recon': recon,
            'mu': mu,
            'logvar': logvar,
            'prior_mu': prior_mu,
            'prior_logvar': prior_logvar,
            'z': z
        }

    def encode(
        self,
        current_tactile: torch.Tensor,
        visual_history: torch.Tensor,
        tactile_history: torch.Tensor,
        action_history: torch.Tensor
    ) -> torch.Tensor:
        """Encode to latent space (for downstream policy use)."""
        mu, logvar = self.encoder(
            current_tactile=current_tactile,
            visual_history=visual_history,
            tactile_history=tactile_history,
            action_history=action_history
        )
        # Use mean for deterministic encoding
        return mu
