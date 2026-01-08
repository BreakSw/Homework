import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import time
import sys

# 固定上下文长度：要和你导出时 dummy seq_len 一致（你当前是 32）
CTX_LEN = 32

MAX_TOKENS = 80
TEMPERATURE = 0.8
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.10
REPEAT_WINDOW = 64


def _pad_or_truncate_to_ctx(input_ids: np.ndarray, pad_id: int) -> tuple[np.ndarray, int]:
    assert input_ids.ndim == 2 and input_ids.shape[0] == 1
    real_len = int(input_ids.shape[1])
    if real_len >= CTX_LEN:
        input_ids = input_ids[:, -CTX_LEN:]
        real_len = CTX_LEN
    else:
        pad = np.full((1, CTX_LEN - real_len), pad_id, dtype=np.int64)
        input_ids = np.concatenate([input_ids.astype(np.int64), pad], axis=1)
    return input_ids.astype(np.int64), real_len


def sample_next_token(
    logits_last: np.ndarray,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    recent_tokens: list[int] | None,
) -> int:
    x = logits_last.astype(np.float64)

    if recent_tokens:
        for t in set(recent_tokens):
            x[t] /= repetition_penalty

    x /= max(float(temperature), 1e-6)

    if top_k is not None and 0 < top_k < x.size:
        idx = np.argpartition(x, -top_k)[-top_k:]
        mask = np.ones_like(x, dtype=bool)
        mask[idx] = False
        x[mask] = -1e10

    x -= x.max()
    p = np.exp(x)
    s = p.sum()
    if s == 0 or not np.isfinite(s):
        return int(np.argmax(logits_last))
    p /= s

    order = np.argsort(-p)
    cdf = np.cumsum(p[order])
    cut = np.searchsorted(cdf, top_p)
    if cut < len(order) - 1:
        drop = order[cut + 1 :]
        p[drop] = 0
        p_sum = p.sum()
        if p_sum > 0:
            p /= p_sum

    return int(np.random.choice(len(p), p=p))


def main():
    mode = (sys.argv[1].strip().lower() if len(sys.argv) > 1 else "int8")
    if mode not in {"fp32", "int8"}:
        print("用法: python chatbot_stu.py fp32   或   python chatbot_stu.py int8")
        return

    model_path = "qwen3_fp32.onnx" if mode == "fp32" else "qwen3_int8.onnx"
    tokenizer_path = "./Qwen3-1.7B"

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id

    input_names = [i.name for i in sess.get_inputs()]
    print(f"[mode={mode}] model={model_path}")
    print("ONNX input names:", input_names)
    print("输入 exit / quit / q 退出。也可 Ctrl+C 退出。")

    while True:
        try:
            q = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if q.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        if not q:
            continue

        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        enc = tokenizer(text, return_tensors="np")
        input_ids = enc["input_ids"].astype(np.int64)

        input_ids, real_len = _pad_or_truncate_to_ctx(input_ids, pad_id)

        print("Qwen:", end=" ", flush=True)

        t0 = time.time()
        n_tokens = 0
        n_chars = 0
        generated: list[int] = []

        for _ in range(MAX_TOKENS):
            ort_inputs = {"input_ids": input_ids}
            logits = sess.run(None, ort_inputs)[0]  # [1, CTX_LEN, V]

            pos = max(real_len - 1, 0)
            logits_last = logits[0, pos, :]

            next_token = sample_next_token(
                logits_last,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                repetition_penalty=REPETITION_PENALTY,
                recent_tokens=generated[-REPEAT_WINDOW:] if REPEAT_WINDOW > 0 else None,
            )
            generated.append(next_token)

            if eos_id is not None and next_token == eos_id:
                break

            word = tokenizer.decode([next_token], skip_special_tokens=True)
            if word:
                print(word, end="", flush=True)
                n_chars += len(word)

            n_tokens += 1

            if real_len < CTX_LEN:
                input_ids[0, real_len] = next_token
                real_len += 1
            else:
                input_ids[:, :-1] = input_ids[:, 1:]
                input_ids[0, -1] = next_token
                real_len = CTX_LEN

        dt = max(time.time() - t0, 1e-6)
        print()
        print(f"[speed] {n_tokens/dt:.2f} token/s, ~{n_chars/dt:.2f} chars/s")


if __name__ == "__main__":
    main()
