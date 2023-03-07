# GPTQ-for-LLaMa
4 bits quantization of [LLaMa](https://arxiv.org/abs/2302.13971) using [GPTQ](https://arxiv.org/abs/2210.17323)
GPTQ is SOTA one-shot weight quantization method

**This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)**

## Result
| Model([LLaMa-7B](https://arxiv.org/abs/2302.13971))      | Bits | group-size | Wikitext2 |   PTB     |    C4   |
| ---------                                                | ---- | ---------- | --------- | --------- | ------- |
| FP16                                                     |  16  |     -      |    5.67   |    8.79   |   7.05  | 
| RTN                                                      |  4   |     -      |    6.28   |    9.68   |   7.70  | 
| [GPTQ](https://arxiv.org/abs/2210.17323)                 |  4   |   1024     |    6.98   |   10.81   |   7.99  |
| [GPTQ](https://arxiv.org/abs/2210.17323)                 |  4   |    64      |    **6.16**   |    **9.66**   |   **7.52**  | 
| RTN                                                      |  3   |     -      |    25.66   |    61.25   |   28.19  | 
| [GPTQ](https://arxiv.org/abs/2210.17323)                 |  3   |    64      |    **12.24**   |    **16.77**   |   **9.55**  | 
| RTN                                                      |  2   |     -      |    101940   |    123128   |   109331  | 
| [GPTQ](https://arxiv.org/abs/2210.17323)                 |  2   |    64      |    **75.28**   |    **241.18**   |   **60.79**  |

| Model([LLaMa-13B](https://arxiv.org/abs/2302.13971))     | Bits | group-size | Wikitext2 |   PTB     |    C4   |
| ---------                                                | ---- | ---------- | --------- | --------- | ------- |
| FP16                                                     |  16  |     -      |    5.08   |    8.06   |   6.58  | 
| RTN                                                      |  4   |     -      |    5.52   |    8.62   |   6.96  | 
| [GPTQ](https://arxiv.org/abs/2210.17323)                 |  4   |    64      |    **5.18**   |    **8.18**   |   **6.66**  |
| RTN                                                      |  3   |     -      |    11.41   |    21.21   |   13.20  | 
| [GPTQ](https://arxiv.org/abs/2210.17323)                 |  3   |    64      |    **5.50**   |    **8.60**   |   **7.00**  |

Quantizing the model requires a large amount of CPU memory. For example, quantizing a LLaMa-13b model requires 32gb, and LLaMa-33b requires more memory than 64gb.

According to [the case for 4-bit precision paper](https://arxiv.org/abs/2212.09720) and [GPTQ paper](https://arxiv.org/abs/2210.17323), a lower group-size achieves a lower ppl(perplexity). Therefore, a group-size lower than 128 is recommended.

Depending on the GPUs/drivers, there may be a difference in performance, which decreases as the model size increases.(https://github.com/IST-DASLab/gptq/issues/1)

According to [GPTQ paper](https://arxiv.org/abs/2210.17323), As the size of the model increases, the difference in performance between FP16 and GPTQ decreases.

## Dependencies

* `torch`: tested on v1.12.1+cu113
* `transformers`: [tested on v4.27.0.dev0(required)](https://github.com/zphang/transformers/tree/llama_push)
* `datasets`: tested on v2.10.1

All experiments were run on a single NVIDIA RTX3090.

## Language Generation

### LLaMa
```
# Compute full precision (FP16) results
CUDA_VISIBLE_DEVICES=0 python llama.py decapoda-research/llama-7b-hf c4
# Run RTN baseline and compute results
CUDA_VISIBLE_DEVICES=0 python llama.py decapoda-research/llama-7b-hf c4 --wbits 4 --nearest
# Run GPTQ and compute results
CUDA_VISIBLE_DEVICES=0 python llama.py decapoda-research/llama-7b-hf c4 --wbits 4 --groupsize 64
````

To run other LLaMa models replace `llama-7b-hf` with one of: `llama-13b-hf`, `llama-30b-hf`, `llama-65b-hf`.

## ZeroShot

See `zeroShot/` folder.

## 4-bit CUDA Kernels 
```
# Install kernels
python setup_cuda.py install

# Benchmark performance for FC2 layer of OPT-175B
CUDA_VISIBLE_DEVICES=0 python test_kernel.py

# Benchmark language generation with 4-bit LLaMa-7B:

# Save compressed model
CUDA_VISIBLE_DEVICES=0 python llama.py decapoda-research/llama-7b-hf c4 --wbits 4 --save llama7b-4bit.pt
# Benchmark generating a 128 token sequence with the saved model
CUDA_VISIBLE_DEVICES=0 python llama.py decapoda-research/llama-7b-hf c4 --load llama7b-34it.pt --benchmark 128
# Benchmark FP16 baseline, note that the model will be split across all listed GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python llama.py decapoda-research/llama-7b-hf c4 --benchmark 128
```

According to the [paper](https://arxiv.org/abs/2212.09720), 4-bit quantization is the most efficient in zero-shot inference, so i implement 4-bit instead of 3-bit.

Please note that [GPTQ](https://github.com/IST-DASLab/gptq) 4-bit kernels are currently only optimized for OPT-175B running on 1xA100 or 2xA6000 and may thus yield suboptimal performance on smaller models or on other GPUs.

# Acknowledgements
This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)
Thanks to Meta AI for releasing [LLaMa](https://arxiv.org/abs/2302.13971), a powerful LLM.
