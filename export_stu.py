import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

# ================= TODO 1: 实现掩码补丁 =================
# 根据 input_shape 生成 causal mask:
# - 形状 (batch, 1, seq_len, seq_len)
# - 上三角（未来位置）为 torch.finfo(dtype).min
# - 其余为 0
def mask_patch(*args, **kwargs):
    # 1) 解析 input_shape (batch, seq_len)
    bsz, seq_len = 1, 32  # 默认值
    input_shape = kwargs.get("input_shape", None)

    if input_shape is None:
        # 尝试从 kwargs 里拿 input_ids 的形状
        inp = kwargs.get("input_ids", None)
        if isinstance(inp, torch.Tensor) and inp.ndim >= 2:
            input_shape = inp.shape

    if input_shape is None:
        # 再兜底：从 args 里找 tensor 或 (bsz, seq) tuple
        for a in args:
            if isinstance(a, torch.Tensor) and a.ndim >= 2:
                input_shape = a.shape
                break
            if isinstance(a, (tuple, list)) and len(a) >= 2 and all(isinstance(x, int) for x in a[:2]):
                input_shape = a
                break

    if input_shape is not None:
        bsz = int(input_shape[0])
        seq_len = int(input_shape[1])

    dtype = kwargs.get("dtype", torch.float32)
    device = kwargs.get("device", torch.device("cpu"))

    # 2) 生成掩码：上三角(不含对角线)为 -inf，其余为 0
    neg_inf = torch.finfo(dtype).min
    m2d = torch.full((seq_len, seq_len), neg_inf, dtype=dtype, device=device)
    m2d = torch.triu(m2d, diagonal=1)  # 对角线及以下为 0；上三角为 -inf

    # 3) (seq, seq) -> (bsz, 1, seq, seq)
    mask = m2d.unsqueeze(0).unsqueeze(1).expand(bsz, 1, seq_len, seq_len).contiguous()
    return mask


# 应用补丁
transformers.masking_utils.create_causal_mask = mask_patch
print(">>> [Patch Applied] 已应用掩码补丁")


# ================= TODO 2: 实现模型包装器 (Wrapper) =================
# 关键：use_cache=False + 只返回 logits（Tensor）
class Qwen3ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,        # ✅ 必须关 cache（避免 DynamicCache/复杂对象）
            return_dict=False       # ✅ 返回 tuple，避免 ModelOutput 的复杂结构
        )
        logits = outputs[0]         # ✅ 第一个就是 logits: [B, T, V]
        return logits


# ================= 主程序 =================
model_path = "./Qwen3-1.7B"
output_file = "qwen3_fp32.onnx"

print(f"--- Loading Model ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    base_model.eval()
except Exception as e:
    print(f"Error: {e}")
    exit(1)

model_wrapper = Qwen3ONNXWrapper(base_model)

# 构造虚拟输入
dummy_input_ids = torch.ones((1, 32), dtype=torch.long)
dummy_attention_mask = torch.ones((1, 32), dtype=torch.long)

print(f"--- Exporting to {output_file} ---")

# ================= TODO 3: 配置导出参数 =================
with torch.no_grad():
    torch.onnx.export(
        model_wrapper,
        (dummy_input_ids, dummy_attention_mask),
        output_file,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],

        # ✅ batch(dim0) / seq(dim1) 动态
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },

        # ✅ 建议用 18，避免“先用18再降级到14”带来的兼容坑
        opset_version=18,
        do_constant_folding=True,

        # ✅ 关闭新版 Dynamo 导出器（你之前导出空图/输出丢失，dynamo 路径很容易踩坑）
        dynamo=False,

        # （可选但很建议）大模型权重用外部数据格式，避免 >2GB protobuf 限制
        external_data=True,
    )

print(f"✅ Export Success!")
