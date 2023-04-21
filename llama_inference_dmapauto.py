import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *

import share_tensors_across_processes

from transformers import AutoTokenizer

import accelerate

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map='auto')
    model.seqlen = 2048
    return model

def load_quant(model, checkpoint, wbits, groupsize, device_map):
    from transformers import LlamaConfig, LlamaForCausalLM 
    config = LlamaConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    with accelerate.init_empty_weights():
        model = LlamaForCausalLM(config)
        torch.set_default_dtype(torch.float)
        model = model.eval()
        layers = find_layers(model)
        for name in ['lm_head']:
            if name in layers:
                del layers[name]
        make_quant(model, layers, wbits, groupsize)

    print('Loading model ...')
    model = accelerate.load_checkpoint_and_dispatch(model, checkpoint, device_map=device_map, no_split_module_classes=['LlamaDecoderLayer'])
    model.seqlen = 2048
    print('Done.')

    return model

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
        '--min_length', type=int, default=10,
        help='The minimum length of the sequence to be generated.'
    )
    
    parser.add_argument(
        '--max_length', type=int, default=50,
        help='The maximum length of the sequence to be generated.'
    )
    
    parser.add_argument(
        '--top_p', type=float , default=0.95,
        help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.'
    )
    
    parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='The value used to module the next token probabilities.'
    )

    parser.add_argument(
        '--do_sample', action='store_true',
        help='Perform multinomial sampling (slow) to produce varied output.'
    )

    parser.add_argument(
        '--enable_eos_token', action='store_true',
        help='Check for the completion token every forward pass: https://github.com/huggingface/transformers/pull/22875'
    )

    parser.add_argument(
        '--device_map', type=str, default='auto',
        help='The device_map used to load the model when using accelerate. Default is "auto".'
    )

    parser.add_argument(
        '--keep_alive', action='store_true',
        help='Keep the process alive so others can share the loaded model.'
    )

    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()
    
    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize, args.device_map)
    else:
        model = get_llama(args.model)
        model.eval()
        
    if args.text is not None:
        DEV = next(model.parameters()).device
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        input_ids = tokenizer.encode(args.text, return_tensors="pt").to(DEV)

        if not args.enable_eos_token:
            model.config.eos_token_id = None
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                do_sample=args.do_sample,
                min_length=args.min_length,
                max_length=args.max_length,
                top_p=args.top_p,
                temperature=args.temperature,
            )
        print(tokenizer.decode([el.item() for el in generated_ids[0]]))

    if args.keep_alive:
        print('Keeping process alive to reference model memory.')
        print('Further processes should launch faster.')
        while True:
            time.sleep(60)

