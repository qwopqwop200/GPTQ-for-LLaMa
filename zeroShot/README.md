This folder contains code to reproduce the FewShot tasks. We follow the structure of 
[this](https://github.com/EleutherAI/lm-evaluation-harness) repository for implementing 
our tasks and the evaluation framework.

We implement the following tasks:
- [x] LAMBADA
- [x] PIQA
- [x] ARC-easy
- [x] ARC-challenge
- [x] COPA
- [x] WSC
- [x] RTE
- [x] CB
- [x] StoryCloze-2018


To add new tasks, please follow [this](https://github.com/EleutherAI/lm-evaluation-harness#code-structure) 
instruction.

## Dependencies

* `torch`: tested on v1.10.1+cu111
* `transformers`: tested on v4.21.2
* `datasets`: tested on v1.17.0
* `sacrebleu`: tested on v2.3.1
* `scikit-learn`: tested on v1.0.2

All experiments were run on a single 80GB NVIDIA A100. However, most experiments will work on a GPU with a lot less memory as well.

# Usage

To use the code, you need to simply run the following command:

```bash 
python3 main.py  <model_name> <calibration_dataset> --task <task_name> --num_fewshot <num_fewshot> 
```

### Example: PIQA

To run `OPT` on the PIQA task, you need to run the following command:
```
# Compute full precision (FP16) results 
CUDA_VISIBLE_DEVICES=0 python main.py facebook/opt-125m c4 --task piqa
# Run RTN baseline and compute results
CUDA_VISIBLE_DEVICES=0 python main.py facebook/opt-125m c4 --wbits 4 --nearest --task piqa
# Run GPTQ and compute results
CUDA_VISIBLE_DEVICES=0 python main.py facebook/opt-125m c4 --wbits 4 --task piqa
````

To run other OPT models replace `opt-125m` with one of: `opt-350m`, `opt-1.3b`, `opt-2.7b`, `opt-6.7b`, `opt-13b`, `opt-66b`.
For 175B you must request access from Meta and then convert it to a local HuggingFace checkpoint using their scripts in `metaseq`.
Once you have such a checkpoint, simply pass its path instead of `facebook/opt-125m`. 


To run `BLOOM` models, you need to run the following command:

```
# Compute full precision (FP16) results
CUDA_VISIBLE_DEVICES=0 python main.py bigscience/bloom-560m c4 --task piqa
# Run RTN baseline and compute results
CUDA_VISIBLE_DEVICES=0 python main.py bigscience/bloom-560m c4 --wbits 4 --nearest --task piqa
# Run GPTQ and compute results
CUDA_VISIBLE_DEVICES=0 python main.py bigscience/bloom-560m c4 --wbits 4 --task piqa
````

To run other BLOOM models replace `bloom-560m` with one of: `bloom-1b1`, `bloom-1b7`, `bloom-3b`, `bloom-7b1`, `bloom`.

