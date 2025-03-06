import torch
import torch.nn as nn 
import vision_transformer

class AttentionRollout(nn.Module):
    def __init__(self, checkpoint_path, arch="vit_small", patch_size=16, drop_path=0, head_fusion="min"):
        super().__init__()
        
        # Initialize the ViT encoder with minimal necessary arguments
        self.encoder = vision_transformer.__dict__[arch](
            patch_size=patch_size,
            drop_path_rate=drop_path,
            use_mean_pooling=False,
            )
        self.head_fusion = head_fusion
        
        # Load pretrained weights from checkpoint
        self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint = checkpoint["model"]['student'] if "model" in checkpoint else checkpoint

        
        self.encoder.load_state_dict(checkpoint, strict=False)
        print(f"Model weights loaded from {checkpoint_path}")


    def aggregate_heads(self, attention_map):
        if self.head_fusion == "mean":
            return attention_map.mean(dim=1)
        elif self.head_fusion == "max":
            return attention_map.max(dim=1)[0]
        elif self.head_fusion == "min":
            return attention_map.min(dim=1)[0]
        else:
            raise ValueError(f"Invalid head fusion method: {self.head_fusion}")

    def compute_attention_rollout(self,x):
        attention_rollout = None
        attention_maps = self.encoder.get_attention_maps(x)
        for attn_map in reversed(attention_maps):
            fused_attentions = self.aggregate_heads(attn_map)
            I = torch.eye(fused_attentions.size(-1), device=x.device)
            normalized_attention = (fused_attentions + I) / 2
            normalized_attention /= normalized_attention.sum(dim=-1, keepdim=True)

            if attention_rollout == None:
                attention_rollout = normalized_attention
            else:
                attention_rollout = attention_rollout @ normalized_attention

        return attention_rollout[:, 0, :] # type: ignore


    def forward(self, x):
        # Compute attention rollout and mask
        attention_rollout = self.compute_attention_rollout(x)

        return attention_rollout