import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


MODEL_CANDIDATES = [
    "/nfsdat1/home/rfchenslm/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    "/nfsdat1/home/rfchenslm/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
]


def _has_weights(path: str) -> bool:
    index_file = f"{path}/model.safetensors.index.json"
    first_shard = f"{path}/model-00001-of-00004.safetensors"
    return os.path.exists(index_file) and os.path.exists(first_shard)


def _pick_model_path() -> str:
    for candidate in MODEL_CANDIDATES:
        if _has_weights(candidate):
            return candidate

    raise FileNotFoundError(
        "未找到可用本地权重分片。请确认模型目录下存在 model-00001-of-00004.safetensors 等文件。"
    )


class UpMLP(nn.Module):
    """
    把每个机器人的 64 维 v 映射成 n 个 4096 维向量。

    输入:
        v: [B, R, 64]

    输出:
        Z: [B, R, n, 4096]
    """

    def __init__(
        self,
        v_dim: int = 64,
        hidden_dim: int = 512,
        n_tokens: int = 8,
        llm_dim: int = 4096,
        output_scale: float = 0.05,
    ):
        super().__init__()

        self.n_tokens = n_tokens
        self.llm_dim = llm_dim
        self.output_scale = output_scale

        out_dim = n_tokens * llm_dim

        self.net = nn.Sequential(
            nn.Linear(v_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        v: [B, R, 64]
        """

        B, R, _ = v.shape

        z = self.net(v)
        # z: [B, R, n * 4096]

        # 固定小尺度缩放，避免随机初始化阶段输出幅度过大
        z = self.output_scale * z

        z = z.reshape(B, R, self.n_tokens, self.llm_dim)
        # z: [B, R, n, 4096]

        return z


class DownMLP(nn.Module):
    """
    把每个机器人的 n 个 4096 维 hidden vectors 映射回 64 维 v。

    输入:
        H: [B, R, n, 4096]

    输出:
        v_hat: [B, R, 64]
    """

    def __init__(
        self,
        v_dim: int = 64,
        hidden_dim: int = 512,
        n_tokens: int = 8,
        llm_dim: int = 4096,
    ):
        super().__init__()

        self.n_tokens = n_tokens
        self.llm_dim = llm_dim

        in_dim = n_tokens * llm_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, v_dim),
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        H: [B, R, n, 4096]
        """

        B, R, n, D = H.shape

        assert n == self.n_tokens, (
            f"DownMLP 收到的 n={n}, 但初始化时 n_tokens={self.n_tokens}"
        )
        assert D == self.llm_dim, (
            f"DownMLP 收到的 hidden_dim={D}, 但初始化时 llm_dim={self.llm_dim}"
        )

        h = H.reshape(B, R, n * D)
        # h: [B, R, n * 4096]

        v_hat = self.net(h)
        # v_hat: [B, R, 64]

        return v_hat


def main():
    model_path = _pick_model_path()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16
    if device == "cpu":
        dtype = torch.float32

    print("Using model path:", model_path)
    print("Using device:", device)
    print("Using dtype:", dtype)

    print("Loading config...")
    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
    )

    print("hidden_size:", config.hidden_size)
    print("num_hidden_layers:", config.num_hidden_layers)

    assert config.hidden_size == 4096, (
        f"模型 hidden_size={config.hidden_size}, 但你的输入维度是 4096，不匹配"
    )

    print("Loading model...")
    llm = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    # 冻结大模型参数
    for p in llm.parameters():
        p.requires_grad = False

    llm.eval()

    # 参数设置
    B = int(os.getenv("B", "1"))
    R = int(os.getenv("R", "112"))
    n = int(os.getenv("N", "8"))
    v_dim = 64
    D = 4096

    print("B:", B)
    print("R:", R)
    print("n_tokens:", n)
    print("llm_dim:", D)

    # MLP_up 输出缩放系数
    output_scale = float(os.getenv("OUTPUT_SCALE", "0.05"))
    print("MLP_up output_scale:", output_scale)

    # 定义前置 MLP：64 -> n*4096
    mlp_up = UpMLP(
        v_dim=v_dim,
        hidden_dim=512,
        n_tokens=n,
        llm_dim=D,
        output_scale=output_scale,
    ).to(device=device, dtype=dtype)

    # 定义后置 MLP：n*4096 -> 64
    mlp_down = DownMLP(
        v_dim=v_dim,
        hidden_dim=512,
        n_tokens=n,
        llm_dim=D,
    ).to(device=device, dtype=dtype)

    mlp_up.eval()
    mlp_down.eval()

    # 这里先用随机 v 模拟真实机器人通信向量
    # 后面接入你的系统时，这个 v 应该来自真实模型输出
    v = torch.randn(B, R, v_dim, device=device, dtype=dtype)
    print("Input v:", v.shape)

    # v: [B,112,64] -> Z: [B,112,n,4096]
    with torch.no_grad():
        Z = mlp_up(v)

    print("After MLP_up, Z:", Z.shape)

    # [B,112,n,4096] -> [B,112*n,4096]
    Z_seq = Z.reshape(B, R * n, D)

    print("Input to LLaMA:", Z_seq.shape)

    attention_mask = torch.ones(
        B,
        R * n,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        outputs = llm(
            inputs_embeds=Z_seq,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    # 取最后一层 hidden state
    H_seq = outputs.hidden_states[-1]

    print("Output from LLaMA:", H_seq.shape)

    # [B,112*n,4096] -> [B,112,n,4096]
    H = H_seq.reshape(B, R, n, D)

    print("Reshaped H:", H.shape)

    # H: [B,112,n,4096] -> v_hat: [B,112,64]
    with torch.no_grad():
        v_hat = mlp_down(H)

    print("After MLP_down, v_hat:", v_hat.shape)

    # 这里只是测试流程，所以简单打印一个重构误差。
    # 现在 MLP 是随机初始化的，这个 loss 没有实际意义，只是确认形状可以算。
    recon_loss = torch.mean((v_hat.float() - v.float()) ** 2)
    print("Dummy recon loss:", recon_loss.item())


if __name__ == "__main__":
    main()