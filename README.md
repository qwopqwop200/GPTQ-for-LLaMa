# GPTQ-for-LLaMa
4 bits quantization of [LLaMa](https://arxiv.org/abs/2302.13971) using [GPTQ](https://arxiv.org/abs/2210.17323)

GPTQ is SOTA one-shot weight quantization method

**This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)**

## New Features
Changed to support new features proposed by [GPTQ](https://github.com/IST-DASLab/gptq#new-features).

* Slightly adjusted preprocessing of C4 and PTB for more realistic evaluations (used in our updated results); can be activated via the flag --new-eval.
* Optimized cuda kernels, which are considerably faster especially on the A100, e.g. 1.9x -> 3.25x generation speedup for OPT-175B; can be activated via --faster-kernel.
* two new tricks:--act-order (quantizing columns in order of decreasing activation size) and --true-sequential (performing sequential quantization even within a single Transformer block). Those fix GPTQ's strangely bad performance on the 7B model (from 7.15 to 6.09 Wiki2 PPL) and lead to slight improvements on most models/settings in general. 

**Currently, `groupsize` and `act-order` do not work together and you must choose one of them.**

## Result
<details>
<summary>LLaMa-7B(click me)</summary>

| [LLaMa-7B](https://arxiv.org/abs/2302.13971)       | Bits | group-size | memory(MiB) | Wikitext2 | checkpoint size(GB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ------------------- |
| FP16                                               |  16  |     -      |    13940    |    5.68   |         12.5        |
| RTN                                                |  4   |     -      |      -      |    6.29   |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |     -      |     4740    |    6.09   |          3.5        |
| RTN                                                |  3   |     -      |      -      |   25.54   |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |     -      |     3852    |    8.07   |          2.7        |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |    128     |     4116    |    6.61   |          3.0        |

</details>

<details>
<summary>LLaMa-13B</summary>

| [LLaMa-13B](https://arxiv.org/abs/2302.13971)      | Bits | group-size | memory(MiB) | Wikitext2 | checkpoint size(GB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ------------------- |
| FP16                                               |  16  |     -      |     OOM     |    5.09   |         24.2        |
| RTN                                                |  4   |     -      |      -      |    5.53   |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |     -      |     8410    |    5.36   |          6.5        |
| RTN                                                |  3   |     -      |      -      |   11.40   |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |     -      |     6870    |    6.63   |          5.1        |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |    128     |     7277    |    5.62   |          5.4        |

</details>

<details>
<summary>LLaMa-33B</summary>

| [LLaMa-33B](https://arxiv.org/abs/2302.13971)      | Bits | group-size | memory(MiB) | Wikitext2 | checkpoint size(GB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ------------------- |
| FP16                                               |  16  |     -      |     OOM     |    4.10   |         60.5        |
| RTN                                                |  4   |     -      |      -      |    4.54   |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |     -      |    19493    |    4.45   |         15.7        |
| RTN                                                |  3   |     -      |      -      |   14.89   |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |     -      |    15493    |    5.69   |         12.0        |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |    128     |    16566    |    4.80   |         13.0        |

</details>

<details>
<summary>LLaMa-65B</summary>

| [LLaMa-65B](https://arxiv.org/abs/2302.13971)      | Bits | group-size | memory(MiB) | Wikitext2 | checkpoint size(GB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ------------------- |
| FP16                                               |  16  |     -      |     OOM     |    3.53   |         121.0       |
| RTN                                                |  4   |     -      |      -      |    3.92   |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |     -      |     OOM     |    3.84   |         31.1        |
| RTN                                                |  3   |     -      |      -      |   10.59   |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |     -      |     OOM     |    5.04   |         23.6        |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |    128     |     OOM     |    4.17   |         25.6        |
</details>

Quantization requires a large amount of CPU memory. However, the memory required can be reduced by using swap memory.

Depending on the GPUs/drivers, there may be a difference in performance, which decreases as the model size increases.(https://github.com/IST-DASLab/gptq/issues/1)

According to [GPTQ paper](https://arxiv.org/abs/2210.17323), As the size of the model increases, the difference in performance between FP16 and GPTQ decreases.

## Installation
If you don't have [conda](https://docs.conda.io/en/latest/miniconda.html), install it first.
```
conda create --name gptq python=3.9 -y
conda activate gptq
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# Or, if you're having trouble with conda, use pip with python3.9:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa
cd GPTQ-for-LLaMa
pip install -r requirements.txt
python setup_cuda.py install

# Benchmark performance for FC2 layer of LLaMa-7B
CUDA_VISIBLE_DEVICES=0 python test_kernel.py
```
## Dependencies

* `torch`: tested on v2.0.0+cu117
* `transformers`: tested on v4.28.0.dev0
* `datasets`: tested on v2.10.1
* `safetensors`: tested on v0.3.0
* (to run 4-bit kernels: setup for compiling PyTorch CUDA extensions, see also https://pytorch.org/tutorials/advanced/cpp_extension.html, tested on CUDA 11.7)

All experiments were run on a single NVIDIA RTX3090.

# Language Generation
## LLaMa

```
#convert LLaMa to hf
python convert_llama_weights_to_hf.py --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir ./llama-hf

# Benchmark language generation with 4-bit LLaMa-7B:

# Save compressed model
CUDA_VISIBLE_DEVICES=0 python llama.py ./llama-hf/llama-7b c4 --wbits 4 --true-sequential --act-order --save llama7b-4bit.pt
# Or save compressed `.safetensors` model
CUDA_VISIBLE_DEVICES=0 python llama.py ./llama-hf/llama-7b c4 --wbits 4 --true-sequential --act-order --save_safetensors llama7b-4bit-128g.safetensors
# Benchmark generating a 2048 token sequence with the saved model
CUDA_VISIBLE_DEVICES=0 python llama.py ./llama-hf/llama-7b c4 --wbits 4 --load llama7b-4bit-128g.pt --benchmark 2048 --check
# Benchmark FP16 baseline, note that the model will be split across all listed GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python llama.py ./llama-hf/llama-7b c4 --benchmark 2048 --check

# model inference with the saved model
CUDA_VISIBLE_DEVICES=0 python llama_inference.py ./llama-hf/llama-7b --wbits 4 --load llama7b-4bit-128g.pt --text "this is llama"
# model inference with the saved model with offload(This is very slow. This is a simple implementation and could be improved with technologies like flexgen(https://github.com/FMInference/FlexGen).
CUDA_VISIBLE_DEVICES=0 python llama_inference_offload.py ./llama-hf/llama-7b --wbits 4 --load llama7b-4bit-128g.pt --text "this is llama" --pre_layer 16
It takes about 180 seconds to generate 45 tokens(5->50 tokens) on single RTX3090 based on LLaMa-65B. pre_layer is set to 50.
```
CUDA Kernels support 2,3,4,8 bits and Faster CUDA Kernels support 2,3,4 bits.

Basically, 4-bit quantization and 128 groupsize are recommended.

# Acknowledgements
This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)

Thanks to Meta AI for releasing [LLaMa](https://arxiv.org/abs/2302.13971), a powerful LLM.
