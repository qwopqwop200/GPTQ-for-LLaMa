# GPTQ-for-LLaMa
4 bits quantization of LLaMa[https://arxiv.org/abs/2302.13971] using [GPTQ](https://arxiv.org/abs/2210.17323)

**This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)**

## result
| Model([LLaMa-7B](https://arxiv.org/abs/2302.13971))      | Bits | group-size | Wikitext2 |   PTB     |    C4   |
| ---------                                                | ---- | ---------- | --------- | --------- | ------- |
| FP16                                                     |  16  |     -      |    5.67   |    8.79   |   7.05  | 
| RTN                                                      |  4   |     -      |    6.28   |    9.68   |   7.70  | 
| [GPTQ](https://arxiv.org/abs/2210.17323)                 |  4   |   1024     |    6.98   |   10.81   |   7.99  | 
| [GPTQ](https://arxiv.org/abs/2210.17323)                 |  4   |    64      |    6.16   |    9.66   |   7.52  | 

According to [the case for 4-bit precision paper](https://arxiv.org/abs/2212.09720), a lower group-size achieves a lower ppl (perplexity). Therefore, a group-size lower than 128 is recommended.

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

# Acknowledgements
This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)
Thanks to Meta AI for releasing [LLaMa](https://arxiv.org/abs/2302.13971), a powerful LLM.
