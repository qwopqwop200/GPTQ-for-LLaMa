import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *

from transformers import AutoTokenizer

def get_t5(model):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer 
    model_max_length = AutoTokenizer.from_pretrained(model, use_fast=False).model_max_length
    model = AutoModelForSeq2SeqLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model_max_length
    return model

def load_quant(model, checkpoint, wbits, groupsize = -1, warmup_autotune = True):
    from transformers import AutoTokenizer 
    model_max_length = AutoTokenizer.from_pretrained(model, use_fast=False).model_max_length

    from transformers import T5Config, AutoModelForSeq2SeqLM 
    config = T5Config.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForSeq2SeqLM.from_config(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits, groupsize)

    del layers
    
    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict = False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict = False)
    
    model.seqlen = model_max_length
    print('Done.')
    
    return model

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='t5 model to load'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
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
        '--min_length', type=int, default=20,
        help='The minimum length of the sequence to be generated.'
    )
    
    parser.add_argument(
        '--max_length', type=int, default=250,
        help='The maximum length of the sequence to be generated.'
    )
    
    # parser.add_argument(
    #     '--top_p', type=float , default=0.95,
    #     help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.'
    # )
    
    parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='The value used to module the next token probabilities.'
    )

    parser.add_argument(
        '--device', type=int, default=-1,
        help='The device used to load the model when using safetensors. Default device is "cpu" or specify, 0,1,2,3,... for GPU device.'
    )

    parser.add_argument(
        "--repl", action="store_true", required=False,
        help="Run REPL"
    )

    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()
    
    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize, args.device)
    else:
        model = get_t5(args.model)
        model.eval()
        
    model.to(DEV)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def predict_fn(data, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

        # retrieve prompt
        text = data.pop("inputs", data)

        # tokenize prompt and use it (together with other generation parameters) to create the model response
        inputs = tokenizer(text, return_tensors="pt").input_ids.to(DEV)
        outputs = model.generate(inputs, **data)
        
        # return model output and skip special tokens (such as "<s>")
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    data = {
    "inputs": '', 
    "min_length": args.min_length, 
    "max_length": args.max_length,
    "do_sample": True,
    "temperature": args.temperature,
    }

    if args.text:
        data['inputs'] = args.text
        res = predict_fn(data,(model,tokenizer))
        print('Prompt: ', args.text)
        print('Output: ',res)

    # Simple Prompt/Output REPL
    if args.repl:
        input_str = ''
        print('Type quit or exit to exit this loop.')
        while input_str != 'quit' and input_str != 'exit':
            input_str = input('Enter a prompt: ')
            data['inputs'] = input_str
            res = predict_fn(data,(model,tokenizer))
            print('Output: ', res)