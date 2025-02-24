import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn_output, _ = self.attention(q, k, v)
        return self.o_proj(attn_output)

class Expert(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, embed_dim)
        self.w3 = nn.Linear(embed_dim, hidden_dim)
    
    def forward(self, x):
        x = F.gelu(self.w1(x))
        x = F.gelu(self.w2(x))
        return self.w3(x)

class MoELayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(embed_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(embed_dim, num_experts)  # 用于路由
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        gate_logits = self.gate(x)  # (batch, seq_len, num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (batch, seq_len, num_experts)
        
        topk_values, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)  # 选择 top-2
        
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = topk_indices[..., i]
            weight = topk_values[..., i].unsqueeze(-1)  # (batch, seq_len, 1)
            
            expert_outputs = torch.stack([self.experts[j](x) for j in range(self.num_experts)], dim=-1)  # (batch, seq_len, embed_dim, num_experts)
            
            selected_expert_output = torch.gather(expert_outputs, -1, expert_idx.unsqueeze(-1).expand(-1, -1, embed_dim, 1)).squeeze(-1)  # (batch, seq_len, embed_dim)
            
            output += weight * selected_expert_output
        
        return output

class MixtralBlock(nn.Module):
    def __init__(self, embed_dim=4096, num_heads=32, num_experts=8, hidden_dim=14336):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.post_attention_layernorm = nn.LayerNorm(embed_dim)
        self.moe = MoELayer(embed_dim, hidden_dim, num_experts)
    
    def forward(self, x):
        x = self.input_layernorm(x)
        x = x + self.attn(x)  # 残差连接
        x = self.post_attention_layernorm(x)
        x = x + self.moe(x)  # 残差连接
        return x

# 创建随机输入
batch_size = 2
seq_len = 128
embed_dim = 4096

random_input = torch.randn(batch_size, seq_len, embed_dim)

# 初始化 Mixtral Block
mixtral_block = MixtralBlock()
output = mixtral_block(random_input)

print("Output shape:", output.shape)
