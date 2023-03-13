import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *

from transformers import AutoTokenizer
from tqdm import tqdm
import json

DEV = torch.device('cuda:0')


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LLaMAForCausalLM
    model = LLaMAForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


def load_quant(model, checkpoint, wbits):
    from transformers import LLaMAConfig, LLaMAForCausalLM
    config = LLaMAConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LLaMAForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits)

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done.')

    return model


prompt_template = """
Please fill in the blanks. Write A or B as the answer.

Sentence: Sarah was a much better surgeon than Maria so _ always got the harder cases.
Option A: Sarah
Option B: Maria
Answer: A

Sentence: They were worried the wine would ruin the bed and the blanket, but the _ was't ruined.
Option A: blanket
Option B: bed
Answer: B

Sentence: Terry tried to bake the eggplant in the toaster oven but the _ was too big.
Option A: eggplant
Option B: toaster
Answer: A

Sentence: {sentence}
Option A: {option1}
Option B: {option2}
"""


def test_one_sample(sample, model, tokenizer):
    prompt = prompt_template.replace("{sentence}", sample["sentence"])
    prompt = prompt.replace("{option1}", sample["option1"])
    prompt = prompt.replace("{option2}", sample["option2"])

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEV)

    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        min_length=args.min_length,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=3,
        top_k=35

    )
    answer = tokenizer.decode([el.item() for el in generated_ids[0]])[-1]
    # Llama (at least the 7B) won't answer at all with 1 or 2 as option but works with A or B
    if answer=="A":
        answer = "1"
    elif answer=="B":
        answer = "2"

    success = answer == sample["answer"]

    print("sentence: " + sample["sentence"])
    print("Option 1: " + sample["option1"])
    print("Option 2: " + sample["option2"])
    print("Expected answer: " + sample["answer"])
    print("Actual answer: "+answer)
    print("PASS" if success else "FAIL")
    return success


def eval_winogbrande(model, tokenizer):
    samples = []
    with open("winogrande/train_debiased.jsonl") as dataset:
        for line in dataset:
            sample = json.loads(line)
            samples.append(sample)
    total = len(samples)
    correct = 0


    for i, sample in enumerate(tqdm(samples, "Testing")):
        print("-----------------------------------")
        result = test_one_sample(sample, model, tokenizer)
        if result:
            correct+=1
        print("Rate: " + str(correct/(i+1)*100)+"%")



if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='llama model to load'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )

    parser.add_argument(
        '--text', type=str,
        help='input text'
    )

    parser.add_argument(
        '--min_length', type=int, default=10,
        help='The minimum length of the sequence to be generated.'
    )

    parser.add_argument(
        '--max_length', type=int, default=50,
        help='The maximum length of the sequence to be generated.'
    )

    parser.add_argument(
        '--top_p', type=float, default=0.95,
        help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.'
    )

    parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='The value used to module the next token probabilities.'
    )

    args = parser.parse_args()

    if args.load:
        model = load_quant(args.model, args.load, args.wbits)
    else:
        model = get_llama(args.model)
        model.eval()

    model.to(DEV)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    with torch.no_grad():
        eval_winogbrande(model, tokenizer)

    print()
