import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, embed_dim)
    
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
        self.gate = nn.Linear(embed_dim, num_experts)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        topk_values, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        
        output = torch.zeros_like(x)
        expert_outputs = torch.stack([self.experts[j](x) for j in range(self.num_experts)], dim=-1)
        
        for i in range(self.top_k):
            expert_idx = topk_indices[..., i].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, embed_dim, 1)
            selected_expert_output = torch.gather(expert_outputs, -1, expert_idx).squeeze(-1)
            weight = topk_values[..., i].unsqueeze(-1)
            output += weight * selected_expert_output
        
        return output

# 创建随机输入
batch_size = 2
seq_len = 128
embed_dim = 4096

random_input = torch.randn(batch_size, seq_len, embed_dim)

# 初始化 MoE 层
moe_layer = MoELayer(embed_dim=4096, hidden_dim=14336, num_experts=8)
output = moe_layer(random_input)

print("Output shape:", output.shape)
