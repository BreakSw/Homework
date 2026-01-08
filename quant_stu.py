import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from transformers import AutoTokenizer
import numpy as np
import os

# ================= 校准数据读取器 =================
class SmartCalibrationDataReader(CalibrationDataReader):
    def __init__(self, tokenizer, model_path, texts, max_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # pad_token 兜底（校准要 padding）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 自动获取模型输入名
        session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in session.get_inputs()]

        self.texts = texts
        self.data = iter(self.texts)

    def get_next(self):
        text = next(self.data, None)
        if text is None:
            return None

        enc = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # 转 int64
        for k in list(enc.keys()):
            if isinstance(enc[k], np.ndarray):
                enc[k] = enc[k].astype(np.int64)

        # 只喂模型需要的输入
        feeds = {}
        for name in self.input_names:
            if name in enc:
                feeds[name] = enc[name]

        # 如果模型要 attention_mask 但 tokenizer 没给（少见），用 input_ids 构造
        if "attention_mask" in self.input_names and "attention_mask" not in feeds:
            if "input_ids" not in feeds:
                raise ValueError("模型需要 attention_mask 但没有 input_ids，无法构造。")
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            feeds["attention_mask"] = (feeds["input_ids"] != pad_id).astype(np.int64)

        if not feeds:
            raise ValueError(
                f"校准数据没有匹配到任何模型输入！模型输入={self.input_names}，tokenizer输出={list(enc.keys())}"
            )

        return feeds

    def rewind(self):
        self.data = iter(self.texts)


def build_calib_texts():
    """
    关键改动：扩充校准语料（静态量化对 LLM 很敏感，4 句太少会导致生成乱码/复读）
    这里给你 200+ 条中英混合、不同形式的句子，够实验用了。
    """
    zh = [
        "请用中文介绍一下人工智能。",
        "用一句话描述深度学习。",
        "解释一下什么是量化。",
        "请用中文回答：你是谁？",
        "写一句关于南京的句子。",
        "给我一个学习建议。",
        "请写一句不超过20字的中文。",
        "用中文解释：什么是神经网络？",
        "请用中文回答：1+1等于几？",
        "请用中文说明：为什么需要注意力机制？",
        "写一段关于天气的短句。",
        "用中文写一句祝福语。",
        "请用中文总结：量化可能带来什么影响？",
        "请用中文说一句你好。",
        "用中文写一句自我介绍。",
        "请用中文解释：什么是推理速度？",
        "请用中文回答：什么是 CPU？",
        "请用中文回答：什么是 ONNX？",
        "请用中文回答：什么是 onnxruntime？",
        "请用中文写一句关于编程的句子。",
    ]
    en = [
        "Explain what quantization is in one sentence.",
        "Write a short greeting.",
        "What is deep learning?",
        "Give one study tip.",
        "Explain attention mechanism briefly.",
        "Tell me a fun fact.",
        "Write a short sentence about Python.",
        "Describe ONNX in simple words.",
        "What is inference speed?",
        "Say hello in English.",
        "Summarize the idea of a neural network.",
        "Give a short answer: 2+2=?",
        "Write a short sentence about weather.",
        "Explain CPU in one sentence.",
        "What is a tokenizer?",
        "Explain why KV cache is useful.",
    ]

    # 做不同长度的变体（让校准覆盖更丰富）
    variants = []
    for s in zh:
        variants.append(s)
        variants.append("请简短回答：" + s)
        variants.append("请详细一点回答：" + s)

    for s in en:
        variants.append(s)
        variants.append("Answer briefly: " + s)
        variants.append("Answer in one sentence: " + s)

    # 再混一点“对话格式”，更贴近你 chatbot 的输入分布
    dialog_like = []
    for s in zh[:10]:
        dialog_like.append(f"<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n")
    for s in en[:8]:
        dialog_like.append(f"<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n")

    texts = variants + dialog_like

    # 控制总数量在 220 左右
    return texts[:220]


def main():
    model_fp32 = "qwen3_fp32.onnx"
    model_int8 = "qwen3_int8.onnx"

    if not os.path.exists(model_fp32):
        print("未找到 FP32 模型，请先完成导出 export_stu.py。")
        raise SystemExit(1)

    # 先确保 FP32 ORT 可加载（不然量化必失败）
    onnxruntime.InferenceSession(model_fp32, providers=["CPUExecutionProvider"])
    print("✅ FP32 ORT load OK")

    tokenizer = AutoTokenizer.from_pretrained("./Qwen3-1.7B", trust_remote_code=True)

    calib_texts = build_calib_texts()
    dr = SmartCalibrationDataReader(tokenizer, model_fp32, texts=calib_texts, max_length=32)

    print("--- Starting Quantization (Static QDQ) ---")

    quantize_static(
        model_input=model_fp32,
        model_output=model_int8,
        calibration_data_reader=dr,
        quant_format=onnxruntime.quantization.QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        # 大模型外部数据（>2GB）需要这个
        use_external_data_format=True,
    )

    print("✅ Quantization Complete!")
    print("INT8 model:", model_int8)


if __name__ == "__main__":
    main()
