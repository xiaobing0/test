import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

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
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, embed_dim)  # 修正: 14336 -> 4096
    
    def forward(self, x):
        x = F.gelu(self.w1(x))
        x = F.gelu(self.w2(x))
        return self.w3(x)  # 修正: 确保输出维度正确

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
        expert_outputs = torch.stack([self.experts[j](x) for j in range(self.num_experts)], dim=-1)  # (batch, seq_len, embed_dim, num_experts)
        
        for i in range(self.top_k):
            expert_idx = topk_indices[..., i].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, embed_dim, 1)  # (batch, seq_len, embed_dim, 1)
            selected_expert_output = torch.gather(expert_outputs, -1, expert_idx).squeeze(-1)  # (batch, seq_len, embed_dim)
            weight = topk_values[..., i].unsqueeze(-1)  # (batch, seq_len, 1)
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
    

def get_pytorch_model_size(model):
    torch.save(model.state_dict(), "temp_model.pth")
    size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
    os.remove("temp_model.pth")
    return size_mb

# 示例
# model = YourPyTorchModel()
# print(f"Model Size: {get_pytorch_model_size(model):.2f} MB")


# 示例
# model = YourPyTorchModel()
# print(f"Model Size: {get_pytorch_model_size(model):.2f} MB")

torch.cuda.empty_cache()  # 释放显存缓存
torch.cuda.ipc_collect()  # 清理 PyTorch 内存池

# 创建随机输入
batch_size = 1
seq_len = 128
embed_dim = 4096
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random_input = torch.randn(batch_size, seq_len, embed_dim)
mixtral_block = MixtralBlock()

# random_input = torch.randn(batch_size, seq_len, embed_dim, device=device)
# mixtral_block = MixtralBlock().to(device)



t1 = time.time()
output = mixtral_block(random_input)
t2 = time.time()

print("Output shape:", output.shape)
print("Time cost:",t2-t1)
