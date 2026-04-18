import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    """Residual feed-forward block for token-wise feature refinement."""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        """Initialize a residual MLP block."""
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalized residual MLP refinement to the input sequence."""
        return x + self.mlp(self.norm(x))


class GatedResidualMLPBlock(nn.Module):
    """A more expressive residual block with a learned gate."""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        """Initialize a gated residual MLP block."""
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated residual refinement to the input sequence."""
        residual = x
        h = self.norm(x)
        value = F.gelu(self.up_proj(h))
        gate = torch.sigmoid(self.gate_proj(h))
        h = value * gate
        
        h = self.down_proj(h)
        h = self.dropout(h)
        return residual + h


class VisionProjector(nn.Module):
    """Projects vision features into the language model embedding space
    using a deeper residual architecture.
    """
    
    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        hidden_multiplier: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_gated_blocks: bool = True,
    ):
        """Initialize the vision-to-language projection stack."""
        super().__init__()
        
        hidden_dim = language_dim * hidden_multiplier

        self.input_proj = nn.Sequential(
            nn.Linear(vision_dim, language_dim),
            nn.GELU(),
            nn.LayerNorm(language_dim),
        )

        block_cls = GatedResidualMLPBlock if use_gated_blocks else ResidualMLPBlock
        self.blocks = nn.ModuleList([
            block_cls(language_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(language_dim),
            nn.Linear(language_dim, language_dim),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project vision features into the language model embedding space."""
        x = self.input_proj(vision_features)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_proj(x)
        return x
