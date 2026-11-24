"""WikiText-2 perplexity for an MLX model. Fixed slice: first N_WINDOWS windows of SEQ_LEN tokens, no overlap."""
import sys, json, math
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from datasets import load_dataset

SEQ_LEN, N_WINDOWS = 512, 128

def main(path):
    model, tok = load(path)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok.encode("\n\n".join(ds["text"]))[: SEQ_LEN * N_WINDOWS + 1]
    nll, count = 0.0, 0
    for i in range(N_WINDOWS):
        chunk = mx.array(ids[i * SEQ_LEN : (i + 1) * SEQ_LEN + 1])[None]
        logits = model(chunk[:, :-1]).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, chunk[:, 1:], reduction="sum")
        mx.eval(loss)
        nll += loss.item(); count += SEQ_LEN
    print(json.dumps({"model": path, "tokens": count, "ppl": math.exp(nll / count)}))

main(sys.argv[1])
