"""
Hybrid Transformer-GNN model for biomarker estimation
Combines Vision Transformer for spatial features and GNN for structural relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, Optional, Tuple
import math


class PatchEmbedding3D(nn.Module):
    """3D patch embedding for Vision Transformer"""

    def __init__(
        self,
        img_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        in_channels: int = 1,
        embed_dim: int = 256
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.n_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, embed_dim, D', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, n_patches, embed_dim]
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder block"""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class GNNLayer(nn.Module):
    """Graph Neural Network layer with vectorized message passing"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggregation: str = "mean"
    ):
        super().__init__()
        self.aggregation = aggregation
        self.linear = nn.Linear(in_dim, out_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized message passing.
        Args:
            node_features: [N, in_dim]
            edge_index: [2, E]
        """
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            return F.relu(self.linear(node_features))

        source_nodes = node_features[edge_index[0]]  # [E, D]
        target_nodes = node_features[edge_index[1]]  # [E, D]

        # Edge weights
        edge_features = torch.cat([source_nodes, target_nodes], dim=-1)
        edge_weights = torch.sigmoid(self.edge_mlp(edge_features))  # [E, 1]

        # Weighted messages
        messages = source_nodes * edge_weights  # [E, D]

        # Scatter-add aggregation (vectorized)
        aggregated = torch.zeros(
            num_nodes, node_features.shape[-1],
            device=node_features.device, dtype=node_features.dtype
        )
        target_expanded = edge_index[1].unsqueeze(-1).expand_as(messages)
        aggregated.scatter_add_(0, target_expanded, messages)

        if self.aggregation == "mean":
            neighbor_count = torch.zeros(num_nodes, device=node_features.device)
            neighbor_count.scatter_add_(
                0, edge_index[1],
                torch.ones(num_edges, device=node_features.device)
            )
            neighbor_count = torch.clamp(neighbor_count, min=1)
            aggregated = aggregated / neighbor_count.unsqueeze(-1)

        out = self.linear(node_features + aggregated)
        return F.relu(out)


class FusionModule(nn.Module):
    """Fusion module for combining Transformer and GNN features"""

    def __init__(
        self,
        transformer_dim: int,
        gnn_dim: int,
        fusion_dim: int,
        method: str = "attention"
    ):
        super().__init__()
        self.method = method
        combined_dim = transformer_dim + gnn_dim

        if method == "attention":
            self.attention = nn.MultiheadAttention(
                combined_dim,
                num_heads=8,
                batch_first=True
            )
            self.fusion_proj = nn.Linear(combined_dim, fusion_dim)
        elif method == "concat":
            self.fusion_proj = nn.Linear(combined_dim, fusion_dim)
        elif method == "weighted":
            self.weight_transformer = nn.Parameter(torch.tensor(0.5))
            self.weight_gnn = nn.Parameter(torch.tensor(0.5))
            self.proj_t = nn.Linear(transformer_dim, fusion_dim)
            self.proj_g = nn.Linear(gnn_dim, fusion_dim)

    def forward(
        self,
        transformer_features: torch.Tensor,
        gnn_features: torch.Tensor
    ) -> torch.Tensor:
        # Align sequence lengths
        min_len = min(transformer_features.shape[1], gnn_features.shape[1])
        t_feat = transformer_features[:, :min_len, :]
        g_feat = gnn_features[:, :min_len, :]

        if self.method == "attention":
            combined = torch.cat([t_feat, g_feat], dim=-1)
            fused, _ = self.attention(combined, combined, combined)
            return self.fusion_proj(fused)
        elif self.method == "concat":
            combined = torch.cat([t_feat, g_feat], dim=-1)
            return self.fusion_proj(combined)
        elif self.method == "weighted":
            weights = torch.softmax(
                torch.stack([self.weight_transformer, self.weight_gnn]), dim=0
            )
            return weights[0] * self.proj_t(t_feat) + weights[1] * self.proj_g(g_feat)


class HybridTransformerGNN(nn.Module):
    """Main hybrid model combining Transformer and GNN"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_cfg = config['model']['architecture']
        t_cfg = model_cfg['transformer']
        g_cfg = model_cfg['gnn']

        img_size = tuple(config['data']['image_size'])
        patch_size = tuple(t_cfg['patch_size'])

        # Transformer branch
        self.patch_embed = PatchEmbedding3D(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=t_cfg['hidden_dim']
        )

        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(
                embed_dim=t_cfg['hidden_dim'],
                num_heads=t_cfg['num_heads'],
                dropout=t_cfg['dropout']
            )
            for _ in range(t_cfg['num_layers'])
        ])

        # GNN branch
        self.gnn_layers = nn.ModuleList([
            GNNLayer(
                in_dim=g_cfg['hidden_dim'] if i > 0 else t_cfg['hidden_dim'],
                out_dim=g_cfg['hidden_dim'],
                aggregation=g_cfg['aggregation']
            )
            for i in range(g_cfg['num_layers'])
        ])

        # Fusion module
        self.fusion = FusionModule(
            transformer_dim=t_cfg['hidden_dim'],
            gnn_dim=g_cfg['hidden_dim'],
            fusion_dim=model_cfg['fusion']['fusion_dim'],
            method=model_cfg['fusion']['method']
        )

        # Biomarker prediction head
        fusion_dim = model_cfg['fusion']['fusion_dim']
        num_classes = config['biomarkers']['num_classes']
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.n_patches, t_cfg['hidden_dim'])
        )

        # Cache the edge index (built once, reused)
        self._cached_edge_index = None
        self._cached_n_patches = None

        # Whether to use gradient checkpointing
        self.use_checkpointing = config.get('resources', {}).get(
            'enable_gradient_checkpointing', False
        )

    def _build_knn_graph(self, n_patches: int, device: torch.device) -> torch.Tensor:
        """Build k-nearest-neighbor graph based on 3D grid adjacency (26-connectivity)."""
        # Return cached if available
        if self._cached_edge_index is not None and self._cached_n_patches == n_patches:
            return self._cached_edge_index.to(device)

        grid = self.patch_embed.grid_size
        gd, gh, gw = grid

        edge_list = []
        for idx in range(n_patches):
            d = idx // (gh * gw)
            h = (idx % (gh * gw)) // gw
            w = idx % gw

            # 26-connectivity neighborhood
            for dd in [-1, 0, 1]:
                for dh in [-1, 0, 1]:
                    for dw in [-1, 0, 1]:
                        if dd == 0 and dh == 0 and dw == 0:
                            continue
                        nd, nh, nw = d + dd, h + dh, w + dw
                        if 0 <= nd < gd and 0 <= nh < gh and 0 <= nw < gw:
                            neighbor_idx = nd * gh * gw + nh * gw + nw
                            edge_list.append([idx, neighbor_idx])

        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        self._cached_edge_index = edge_index
        self._cached_n_patches = n_patches
        return edge_index.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W] MRI volume
        Returns:
            Biomarker predictions: [B, num_classes]
        """
        B = x.shape[0]

        # Transformer branch
        patch_embeds = self.patch_embed(x)  # [B, N, embed_dim]
        patch_embeds = patch_embeds + self.pos_embed

        transformer_out = patch_embeds
        for layer in self.transformer_layers:
            if self.training and self.use_checkpointing:
                transformer_out = checkpoint(layer, transformer_out, use_reentrant=False)
            else:
                transformer_out = layer(transformer_out)

        # GNN branch: use k-NN spatial graph
        N = patch_embeds.shape[1]
        edge_index = self._build_knn_graph(N, x.device)

        # Process each batch element through GNN
        gnn_outputs = []
        for b in range(B):
            node_features = patch_embeds[b]  # [N, embed_dim]
            gnn_out = node_features
            for gnn_layer in self.gnn_layers:
                gnn_out = gnn_layer(gnn_out, edge_index)
            gnn_outputs.append(gnn_out)

        gnn_out = torch.stack(gnn_outputs, dim=0)  # [B, N, gnn_hidden]

        # Fusion
        fused_features = self.fusion(transformer_out, gnn_out)

        # Global pooling
        pooled = fused_features.mean(dim=1)  # [B, fusion_dim]

        # Prediction
        predictions = self.predictor(pooled)

        return predictions
