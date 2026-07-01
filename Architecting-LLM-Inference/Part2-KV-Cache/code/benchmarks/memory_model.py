"""
KV-cache memory model shared across Part 2 experiments (E01-E03, E09).

Formula (Part 1, Section 3):
    M_kv = 2 x L x B x S x H_kv x D_h x P     (bytes)

    2      -- one tensor for K, one for V
    L      -- number of transformer layers
    B      -- batch size / number of concurrent sequences
    S      -- sequence length (prompt + generated tokens so far)
    H_kv   -- number of KV heads (== H for MHA, < H for GQA, 1 for MQA)
    D_h    -- head dimension
    P      -- bytes per element (2 for fp16/bf16, 1 for int8/fp8, 0.5 for int4)
"""

from dataclasses import dataclass

DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
    "int4": 0.5,
    "int2": 0.25,
}


@dataclass
class ModelConfig:
    name: str
    num_layers: int
    num_q_heads: int
    num_kv_heads: int   # == num_q_heads for MHA; < num_q_heads for GQA; 1 for MQA
    head_dim: int
    hidden_size: int
    attention_type: str  # "MHA" | "GQA" | "MQA"

    @property
    def kv_group_size(self) -> int:
        """How many Q heads share each KV head."""
        return self.num_q_heads // self.num_kv_heads


# A handful of real, open-weight models spanning MHA / GQA / MQA, used
# consistently across E01, E02, E03, and E09 so results are comparable.
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "gpt2": ModelConfig(
        name="gpt2", num_layers=12, num_q_heads=12, num_kv_heads=12,
        head_dim=64, hidden_size=768, attention_type="MHA",
    ),
    "gpt2-xl": ModelConfig(
        name="gpt2-xl", num_layers=48, num_q_heads=25, num_kv_heads=25,
        head_dim=64, hidden_size=1600, attention_type="MHA",
    ),
    "llama2-7b": ModelConfig(
        name="llama2-7b", num_layers=32, num_q_heads=32, num_kv_heads=32,
        head_dim=128, hidden_size=4096, attention_type="MHA",
    ),
    "llama3-8b": ModelConfig(
        name="llama3-8b", num_layers=32, num_q_heads=32, num_kv_heads=8,
        head_dim=128, hidden_size=4096, attention_type="GQA",
    ),
    "qwen2.5-7b": ModelConfig(
        name="qwen2.5-7b", num_layers=28, num_q_heads=28, num_kv_heads=4,
        head_dim=128, hidden_size=3584, attention_type="GQA",
    ),
    "mistral-7b": ModelConfig(
        name="mistral-7b", num_layers=32, num_q_heads=32, num_kv_heads=8,
        head_dim=128, hidden_size=4096, attention_type="GQA",
    ),
    "mqa-hypothetical-7b": ModelConfig(
        name="mqa-hypothetical-7b", num_layers=32, num_q_heads=32, num_kv_heads=1,
        head_dim=128, hidden_size=4096, attention_type="MQA",
    ),
}


def bytes_per_token_per_layer(cfg: ModelConfig, dtype: str = "fp16") -> float:
    """Bytes of KV cache for one token, one layer (both K and V, all KV heads)."""
    return 2 * cfg.num_kv_heads * cfg.head_dim * DTYPE_BYTES[dtype]


def bytes_per_token(cfg: ModelConfig, dtype: str = "fp16") -> float:
    """Bytes of KV cache for one token across all layers."""
    return cfg.num_layers * bytes_per_token_per_layer(cfg, dtype)


def kv_cache_bytes(cfg: ModelConfig, seq_len: int, batch_size: int = 1, dtype: str = "fp16") -> float:
    """
    M_kv = 2 x L x B x S x H_kv x D_h x P

    Full formula, matching Part 1 Section 3 exactly.
    """
    P = DTYPE_BYTES[dtype]
    return 2 * cfg.num_layers * batch_size * seq_len * cfg.num_kv_heads * cfg.head_dim * P


def max_concurrent_sequences(cfg: ModelConfig, seq_len: int, available_bytes: float, dtype: str = "fp16") -> int:
    """How many sequences of `seq_len` tokens fit in `available_bytes` of KV cache."""
    per_seq = kv_cache_bytes(cfg, seq_len, batch_size=1, dtype=dtype)
    if per_seq <= 0:
        return 0
    return int(available_bytes // per_seq)


def human_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


if __name__ == "__main__":
    print("KV-cache bytes/token by model (fp16, seq_len=1 for per-token bytes):\n")
    for name, cfg in MODEL_CONFIGS.items():
        bpt = bytes_per_token(cfg, "fp16")
        print(f"  {name:22s} [{cfg.attention_type:3s}]  "
              f"kv_heads={cfg.num_kv_heads:2d}  group={cfg.kv_group_size:2d}  "
              f"bytes/token={human_bytes(bpt)}")

    print("\nKV-cache size for a 8k-token sequence, batch=1, fp16:\n")
    for name, cfg in MODEL_CONFIGS.items():
        size = kv_cache_bytes(cfg, seq_len=8192, batch_size=1, dtype="fp16")
        print(f"  {name:22s}  {human_bytes(size)}")
