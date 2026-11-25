# mlx-quant-bench

One small model, three precisions, one Mac. Qwen2.5-0.5B-Instruct (apache-2.0, revision 7ae557604adf67be50417f59c2c2f167def9a775) converted with `mlx_lm.convert` to bf16, 8-bit and 4-bit (affine, group size 64), then measured with `mlx_lm.benchmark` and a WikiText-2 perplexity script (`ppl.py`).

Results and raw logs are also on Hugging Face: https://huggingface.co/datasets/trangchang/mlx-quant-bench-results

Hardware: Apple M5, 16 GB, macOS 26.6.2, Python 3.12.10, mlx 0.32.4, mlx-lm 0.32.2. Exact versions in `results/freeze.txt`, machine in `results/env.json`.

## Results (all numbers from `results/`)
| Variant | Peak memory (GB) | WikiText-2 perplexity | Generation tokens/s |
|---|---|---|---|
| bf16 | 1.296 to 1.399 | 19.09 | 31.7 to 41.6 |
| 8-bit | 0.812 to 0.925 | 19.09 | 38.3 to 57.1 |
| 4-bit | 0.529 to 0.678 | 23.29 | 42.2 to 66.3 |

Perplexity: first 65536 tokens of the WikiText-2 raw test split, 128 windows of 512 tokens, no overlap. 8-bit is within 0.01 of bf16. 4-bit is about 22 percent higher.

Speed: `mlx_lm.benchmark -p 512 -g 256`, batch size 1. Two separate runs on the same Mac, same day, gave different absolute speeds (run 1: 5 trials, run 2: 10 trials, see `results/summary.csv`). Inside each run the spread was small, between runs it was about 25 percent. I did not find the cause (possibly memory pressure or power state; not tested). The order bf16 slower than 8-bit slower than 4-bit held in both runs. Treat the absolute speeds as indicative only.

## Limits
One model, one machine, one prompt length, no repeats across days. Perplexity on a fixed slice is not a task benchmark. Nothing here says how other models or other quantization settings behave.

## Reproduce
```
uv venv --python 3.12 && uv pip install mlx==0.32.4 mlx-lm==0.32.2 datasets
mlx_lm.convert --hf-path Qwen/Qwen2.5-0.5B-Instruct --mlx-path models/bf16 --dtype bfloat16
mlx_lm.convert --hf-path Qwen/Qwen2.5-0.5B-Instruct --mlx-path models/q8 -q --q-bits 8 --q-group-size 64 --dtype bfloat16
mlx_lm.convert --hf-path Qwen/Qwen2.5-0.5B-Instruct --mlx-path models/q4 -q --q-bits 4 --q-group-size 64 --dtype bfloat16
mlx_lm.benchmark --model models/q4 -p 512 -g 256 -n 10
python ppl.py models/q4
```
