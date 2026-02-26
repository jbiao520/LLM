"""
KV Cache 优化示例代码

演示 KV Cache 的原理和优化技术。
KV Cache 是 LLM 推理加速的核心技术之一。
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

# ============================================
# 1. KV Cache 基础实现
# ============================================

@dataclass
class KVCache:
    """简单的 KV Cache 实现"""
    key_cache: torch.Tensor    # [batch, num_heads, seq_len, head_dim]
    value_cache: torch.Tensor  # [batch, num_heads, seq_len, head_dim]

    @classmethod
    def create(cls, batch_size: int, num_heads: int, max_seq_len: int, head_dim: int, device: str = "cuda"):
        """创建空的 KV Cache"""
        return cls(
            key_cache=torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device),
            value_cache=torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device),
        )

    def update(self, new_keys: torch.Tensor, new_values: torch.Tensor, start_pos: int):
        """更新 KV Cache"""
        seq_len = new_keys.shape[2]
        self.key_cache[:, :, start_pos:start_pos + seq_len, :] = new_keys
        self.value_cache[:, :, start_pos:start_pos + seq_len, :] = new_values

    def get(self, start_pos: int, end_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定范围的 KV"""
        return (
            self.key_cache[:, :, :end_pos, :],
            self.value_cache[:, :, :end_pos, :]
        )


# ============================================
# 2. 带 KV Cache 的注意力实现
# ============================================

class CachedAttention(nn.Module):
    """带 KV Cache 的注意力层"""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        position: int = 0,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            kv_cache: 之前的 KV Cache
            position: 当前位置
            use_cache: 是否使用 KV Cache
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 投影
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 处理 KV Cache
        if use_cache and kv_cache is not None:
            # 更新缓存
            kv_cache.update(k, v, position)
            # 获取完整的 K, V（包括之前缓存的）
            k, v = kv_cache.get(0, position + seq_len)
        else:
            # 不使用缓存
            pass

        # 计算注意力
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 因果掩码
        if use_cache and kv_cache is not None:
            # 对于缓存模式，只需要掩码新的 query 对 new key
            causal_mask = torch.triu(
                torch.ones(seq_len, position + seq_len, device=hidden_states.device),
                diagonal=position + 1
            ).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        else:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device),
                diagonal=1
            ).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output, kv_cache


# ============================================
# 3. KV Cache 内存分析
# ============================================

def analyze_kv_cache_memory():
    """分析 KV Cache 的内存使用"""

    print("\n" + "="*50)
    print("KV Cache 内存分析")
    print("="*50)

    # 典型配置
    configs = [
        {"model": "LLaMA-2-7B", "layers": 32, "heads": 32, "head_dim": 128},
        {"model": "LLaMA-2-13B", "layers": 40, "heads": 40, "head_dim": 128},
        {"model": "LLaMA-2-70B", "layers": 80, "heads": 64, "head_dim": 128},
    ]

    seq_lengths = [512, 1024, 2048, 4096]

    for config in configs:
        print(f"\n{config['model']}:")
        print("-" * 40)

        for seq_len in seq_lengths:
            # KV Cache 大小计算
            # 2 (K+V) × layers × heads × head_dim × seq_len × 2 bytes (FP16)
            kv_size = 2 * config['layers'] * config['heads'] * config['head_dim'] * seq_len * 2

            print(f"  Seq {seq_len:4d}: {kv_size / 1e9:.2f} GB")


def compare_with_without_cache():
    """对比有无 KV Cache 的计算量"""

    print("\n" + "="*50)
    print("有无 KV Cache 计算量对比")
    print("="*50)

    total_tokens = 100
    hidden_size = 4096

    # 无 KV Cache: 每个位置都要重新计算所有之前的
    flops_no_cache = 0
    for i in range(1, total_tokens + 1):
        # 每次都要计算 i 个位置的注意力
        flops_no_cache += i * hidden_size * 4  # Q, K, V, O 投影

    # 有 KV Cache: 只计算当前位置
    flops_with_cache = total_tokens * hidden_size * 4

    print(f"生成 {total_tokens} tokens:")
    print(f"  无 KV Cache: {flops_no_cache / 1e9:.2f} GFLOPs")
    print(f"  有 KV Cache: {flops_with_cache / 1e9:.2f} GFLOPs")
    print(f"  加速比: {flops_no_cache / flops_with_cache:.1f}x")


# ============================================
# 4. PagedAttention 简化实现
# ============================================

class PagedKVCache:
    """PagedAttention 的简化实现"""

    def __init__(self, num_blocks: int, block_size: int, num_heads: int, head_dim: int):
        """
        Args:
            num_blocks: 总块数
            block_size: 每块的序列长度
            num_heads: 注意力头数
            head_dim: 每头维度
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # 预分配块池
        self.k_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim)
        self.v_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim)

        # 空闲块列表
        self.free_blocks = list(range(num_blocks))

        # 每个请求的块映射
        self.request_blocks = {}  # request_id -> [block_ids]

    def allocate(self, request_id: str, num_tokens: int) -> bool:
        """为请求分配块"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            return False  # 内存不足

        blocks = []
        for _ in range(num_blocks_needed):
            blocks.append(self.free_blocks.pop())

        self.request_blocks[request_id] = blocks
        return True

    def free(self, request_id: str):
        """释放请求的块"""
        if request_id in self.request_blocks:
            self.free_blocks.extend(self.request_blocks[request_id])
            del self.request_blocks[request_id]

    def get_memory_utilization(self) -> float:
        """计算内存利用率"""
        used = self.num_blocks - len(self.free_blocks)
        return used / self.num_blocks


def demo_paged_attention():
    """演示 PagedAttention 的优势"""

    print("\n" + "="*50)
    print("PagedAttention 演示")
    print("="*50)

    cache = PagedKVCache(
        num_blocks=100,
        block_size=16,
        num_heads=32,
        head_dim=128
    )

    # 分配不同长度的请求
    requests = [
        ("req1", 50),   # 需要 4 块
        ("req2", 100),  # 需要 7 块
        ("req3", 30),   # 需要 2 块
    ]

    for req_id, num_tokens in requests:
        success = cache.allocate(req_id, num_tokens)
        print(f"分配 {req_id} ({num_tokens} tokens): {'成功' if success else '失败'}")
        print(f"  内存利用率: {cache.get_memory_utilization():.1%}")

    # 释放一个请求
    print(f"\n释放 req2...")
    cache.free("req2")
    print(f"内存利用率: {cache.get_memory_utilization():.1%}")


# ============================================
# 5. 运行演示
# ============================================

if __name__ == "__main__":
    analyze_kv_cache_memory()
    compare_with_without_cache()
    demo_paged_attention()

    print("\n" + "="*50)
    print("KV Cache 优化示例完成!")
    print("="*50)
