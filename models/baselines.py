"""
Baseline models for comparison
Includes CNN, ViT, GCN, and U-Net architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class BaselineCNN3D(nn.Module):
    """3D CNN baseline model"""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int = 4,
        depth: int = 50
    ):
        super().__init__()

        # Simplified ResNet-like architecture
        self.conv1 = nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(32, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(ResidualBlock3D(in_channels if i == 0 else out_channels, out_channels, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResidualBlock3D(nn.Module):
    """3D Residual block"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BaselineViT3D(nn.Module):
    """3D Vision Transformer baseline"""

    def __init__(
        self,
        img_size: Tuple[int, int, int],
        patch_size: int = 16,
        num_classes: int = 4,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)

        # Patch embedding
        self.patch_embed = nn.Conv3d(
            1, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, D', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Use cls token
        x = self.head(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class BaselineGCN(nn.Module):
    """Graph Convolutional Network baseline"""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 4
    ):
        super().__init__()
        self.num_layers = num_layers

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(nn.Linear(hidden_dim, hidden_dim))

        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        # Global pooling (mean)
        x = x.mean(dim=0, keepdim=True)
        x = self.classifier(x)
        return x


class BaselineUNet3D(nn.Module):
    """3D U-Net baseline for biomarker regression"""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        encoder_depth: int = 4,
        decoder_depth: int = 4
    ):
        super().__init__()
        channels = [32, 64, 128, 256]

        # Encoder
        self.encoder = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = in_channels
        for i in range(encoder_depth):
            out_ch = channels[i] if i < len(channels) else channels[-1]
            self.encoder.append(self._make_encoder_block(in_ch, out_ch))
            if i < encoder_depth - 1:
                self.pools.append(nn.MaxPool3d(2))
            in_ch = out_ch

        # Bottleneck channel count = last encoder output
        bottleneck_ch = channels[min(encoder_depth - 1, len(channels) - 1)]

        # Decoder: (encoder_depth - 1) upsample+decode blocks
        # Connects bottleneck -> skip[-2] -> skip[-3] -> ...
        self.upsamples = nn.ModuleList()
        self.decoder = nn.ModuleList()

        dec_in_ch = bottleneck_ch
        num_decoder_blocks = encoder_depth - 1
        for i in range(num_decoder_blocks):
            # skip connection comes from encoder block at index (encoder_depth - 2 - i)
            skip_idx = encoder_depth - 2 - i
            skip_ch = channels[min(skip_idx, len(channels) - 1)]
            out_ch = skip_ch  # decoder output matches skip channel size

            self.upsamples.append(
                nn.ConvTranspose3d(dec_in_ch, out_ch, kernel_size=2, stride=2)
            )
            # After concat: out_ch (from upsample) + skip_ch (from encoder)
            self.decoder.append(self._make_decoder_block(out_ch + skip_ch, out_ch))
            dec_in_ch = out_ch

        # Final output
        final_ch = channels[0]
        self.final_conv = nn.Conv3d(final_ch, 32, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(32, num_classes)

    def _make_encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder: collect skip connections
        skip_connections = []
        for i, encoder_block in enumerate(self.encoder):
            x = encoder_block(x)
            skip_connections.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)

        # x is now the bottleneck (= skip_connections[-1] after pooling)
        # Decoder: upsample and concatenate with skip connections
        for i, (upsample, decoder_block) in enumerate(zip(self.upsamples, self.decoder)):
            x = upsample(x)

            # Skip from encoder: go from second-to-last backwards
            skip_idx = len(skip_connections) - 2 - i
            if 0 <= skip_idx < len(skip_connections):
                skip = skip_connections[skip_idx]

                # Handle spatial size mismatch
                if x.shape[2:] != skip.shape[2:]:
                    # Pad x to match skip if smaller
                    diff = [s - xs for s, xs in zip(skip.shape[2:], x.shape[2:])]
                    if any(d > 0 for d in diff):
                        pad = []
                        for d in reversed(diff):
                            pad.extend([0, max(0, d)])
                        x = F.pad(x, pad)
                    # Crop to match
                    x = x[:, :, :skip.shape[2], :skip.shape[3], :skip.shape[4]]

                x = torch.cat([x, skip], dim=1)

            x = decoder_block(x)

        # Regression head
        x = self.final_conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
