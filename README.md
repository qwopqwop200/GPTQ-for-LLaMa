# GPTQ-for-LLaMa
4 bits quantization of [LLaMa](https://arxiv.org/abs/2302.13971) using [GPTQ](https://arxiv.org/abs/2210.17323)

GPTQ is SOTA one-shot weight quantization method

**This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)**

## Result
<details>
<summary>LLaMa-7B(click me)</summary>

| [LLaMa-7B](https://arxiv.org/abs/2302.13971)       | Bits | group-size | memory(MiB) | Wikitext2 |   PTB   |    C4   | checkpoint size(GB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ------- | ------- | ------------------- |
| FP16                                               |  16  |     -      |    13940    |    5.67   |   8.79  |   7.05  |         12.5        |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  8   |     -      |     7748    |    5.67   |   8.81  |   7.08  |          6.5        |
| RTN                                                |  4   |     -      |      -      |    6.28   |   9.68  |   7.70  |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |     -      |     4740    |    6.79   |  10.67  |   8.28  |          3.5        |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |    128     |     4891    |    6.26   |   9.71  |   7.60  |          3.7        | 
| RTN                                                |  3   |     -      |      -      |   25.66   |  61.25  |  28.19  |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |     -      |     3852    |   20.86   |  37.54  |  22.19  |          2.7        |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |    128     |     4116    |   10.60   |  14.74  |  10.39  |          3.0        |

</details>

<details>
<summary>LLaMa-13B</summary>

| [LLaMa-13B](https://arxiv.org/abs/2302.13971)      | Bits | group-size | memory(MiB) | Wikitext2 |   PTB   |    C4   | checkpoint size(GB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ------- | ------- | ------------------- |
| FP16                                               |  16  |     -      |     OOM     |    5.08   |   8.06  |   6.58  |         24.2        |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  8   |     -      |    14570    |    5.09   |   8.06  |   6.61  |         12.4        |
| RTN                                                |  4   |     -      |      -      |    5.52   |   8.62  |   6.96  |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |     -      |     8410    |    5.35   |   8.40  |   6.82  |          6.5        |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |    128     |     8747    |    5.21   |   8.20  |   7.60  |          3.7        | 
| RTN                                                |  3   |     -      |      -      |   25.66   |  61.25  |  28.19  |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |     -      |     6870    |    6.77   |  10.29  |   8.34  |          5.1        |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |    128     |     7277    |    5.66   |   8.74  |   6.70  |          5.4        |

</details>

<details>
<summary>LLaMa-33B</summary>

| [LLaMa-33B](https://arxiv.org/abs/2302.13971)      | Bits | group-size | memory(MiB) | Wikitext2 |   PTB   |    C4   | checkpoint size(GB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ------- | ------- | ------------------- |
| FP16                                               |  16  |     -      |     OOM     |     -     |    -    |    -    |         60.5        |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |     -      |    19499    |    4.45   |   7.58  |   6.22  |         17.0        |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |     -      |    15499    |    5.78   |   8.98  |   7.38  |         13.0        |

</details>

Quantizing the model requires a large amount of CPU memory. For example, quantizing a LLaMa-13b model requires 42gb, and LLaMa-33b requires more memory than 64gb. Also, the amount of memory required can be reduced through swap memory.

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
python convert_llama_weights_to_hf.py --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path

# Benchmark language generation with 4-bit LLaMa-7B:

# Save compressed model
CUDA_VISIBLE_DEVICES=0 python llama.py /output/path c4 --wbits 4 --groupsize 128 --save llama7b-4bit.pt
# Or save compressed `.safetensors` model
CUDA_VISIBLE_DEVICES=0 python llama.py /output/path c4 --wbits 4 --groupsize 128 --save_safetensors llama7b-4bit.safetensors
# Benchmark generating a 2048 token sequence with the saved model
CUDA_VISIBLE_DEVICES=0 python llama.py /output/path c4 --wbits 4 --groupsize 128 --load llama7b-4bit.pt --benchmark 2048 --check
# Benchmark FP16 baseline, note that the model will be split across all listed GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python llama.py /output/path c4 --benchmark 2048 --check

# model inference with the saved model
CUDA_VISIBLE_DEVICES=0 python llama_inference.py /output/path --wbits 4 --groupsize 128 --load llama7b-4bit.pt --text "this is llama"
# model inference with the saved model with offload(This is very slow. This is a simple implementation and could be improved with technologies like flexgen(https://github.com/FMInference/FlexGen).
CUDA_VISIBLE_DEVICES=0 python llama_inference_offload.py /output/path --wbits 4 --groupsize 128 --load llama7b-4bit.pt --text "this is llama" --pre_layer 16
It takes about 180 seconds to generate 45 tokens(5->50 tokens) on single RTX3090 based on LLaMa-65B. pre_layer is set to 50.
```
CUDA Kernels support 2,3,4,8 bits.

Basically, 4-bit quantization is recommended.

# Acknowledgements
This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)

Thanks to Meta AI for releasing [LLaMa](https://arxiv.org/abs/2302.13971), a powerful LLM.
