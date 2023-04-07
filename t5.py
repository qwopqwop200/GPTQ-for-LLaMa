import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *

import os
import numpy as np
import pandas as pd

from datasets import load_dataset, get_dataset_config_names, Dataset

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

@torch.no_grad()
def t5_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.decoder.config.use_cache
    model.decoder.config.use_cache = False
    model.config.use_cache = False
    
    layers = model.encoder.block
    model.encoder.embed_tokens = model.encoder.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.encoder.config.d_model), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev)[:,:model.seqlen])
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.encoder.embed_tokens = model.encoder.embed_tokens.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(f'Quantizing {name} in Encoder layer {i+1}/{len(layers)}...')
                scale,zero,g_idx = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order)
                quantizers['encoder.block.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(),scale.cpu(),zero.cpu(),g_idx.cpu())
                gptq[name].free()
                
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        
    model.encoder.final_layer_norm = model.encoder.final_layer_norm.to(dev)
    model.encoder.dropout = model.encoder.dropout.to(dev)
    
    encoder_hidden_states = model.encoder.final_layer_norm(inps)
    encoder_hidden_states = model.encoder.dropout(encoder_hidden_states)
    
    model.encoder.final_layer_norm = model.encoder.final_layer_norm.cpu()
    model.encoder.dropout = model.encoder.dropout.cpu()
    
    layers = model.decoder.block
    model.encoder.embed_tokens = model.encoder.embed_tokens.to(dev)
    model.decoder.embed_tokens = model.decoder.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)
    
    cache = {'i': 0, 'attention_mask': None, 'encoder_attention_mask': None}
    
    inps = torch.zeros((args.nsamples, model.seqlen, model.encoder.config.d_model), dtype=dtype, device=dev)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['encoder_attention_mask'] = kwargs['encoder_attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for j,batch in enumerate(dataloader):
        try:
            model(decoder_input_ids = batch[0].to(dev)[:,model.seqlen:],encoder_outputs = [encoder_hidden_states[j:j+1],])
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    
    model.encoder.embed_tokens = model.encoder.embed_tokens.cpu()
    model.decoder.embed_tokens = model.decoder.embed_tokens.cpu()
    torch.cuda.empty_cache()

    dtype = next(iter(model.parameters())).dtype
    print('Ready.')
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    encoder_attention_mask = cache['encoder_attention_mask']
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                                encoder_hidden_states = encoder_hidden_states[j].unsqueeze(0),
                                encoder_attention_mask = encoder_attention_mask)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(f'Quantizing {name} in Decoder layer {i+1}/{len(layers)}...')
                scale,zero,g_idx = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order)
                quantizers['decoder.block.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(),scale.cpu(),zero.cpu(),g_idx.cpu())
                gptq[name].free()
                
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                            encoder_hidden_states = encoder_hidden_states[j].unsqueeze(0),
                            encoder_attention_mask = encoder_attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        
    model.decoder.config.use_cache = use_cache
    model.config.use_cache = use_cache
    return quantizers

@torch.no_grad()
def t5_nearest_sequential(model, dev):
    layers = model.encoder.block
    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                
                g_idx = torch.zeros(subset[name].in_features,dtype=torch.int32)
                quantizers['encoder.block.%d.%s' % (i, name)] = (quantizer.cpu(),quantizer.scale.cpu(),quantizer.zero.cpu(),g_idx.cpu())
                print(f'Quantizing {name} in Encoder layer {i+1}/{len(layers)}...')
        layer = layers[i].cpu()
    
    layers = model.decoder.block    
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                
                g_idx = torch.zeros(subset[name].in_features,dtype=torch.int32)
                quantizers['decoder.block.%d.%s' % (i, name)] = (quantizer.cpu(),quantizer.scale.cpu(),quantizer.zero.cpu(),g_idx.cpu())
                print(f'Quantizing {name} in Decoder layer {i+1}/{len(layers)}...')
        layer = layers[i].cpu()
    return quantizers

# TODO: perform packing on GPU
def t5_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name],scale,zero,g_idx = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
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
    
    if warmup_autotune:
        autotune_warmup(model)
    model.seqlen = model_max_length
    print('Done.')
    
    return model

# MMLU
subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}
choices = ["A", "B", "C", "D"]

def mmlu_format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def mmlu_gen_prompt(train_df, subject, k=-1):
    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += mmlu_format_example(train_df, i)
    return prompt


@torch.no_grad()
def mmlu_eval(args, subject, model, tokenizer, dev_df, test_df, progress):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain_mmlu
        prompt_end = mmlu_format_example(test_df, i, include_answer=False)
        train_prompt = mmlu_gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        while input_ids.shape[-1] > model.seqlen:
            k -= 1
            train_prompt = mmlu_gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]

        decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.cuda()
        decoder_input_ids = model._shift_right(decoder_input_ids)
        logits = model(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids
        ).logits.flatten().float()

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[0]],
                        logits[tokenizer("B").input_ids[0]],
                        logits[tokenizer("C").input_ids[0]],
                        logits[tokenizer("D").input_ids[0]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}({}/{})".format(acc, subject,progress[0] + 1, progress[1]))

    return cors, acc, all_probs


def mmlu_benchmark(model, tokenizer, args):
    heads_per_gpu = len(model.encoder.block) // args.ngpu
    device_map = {
        gpu: list(
            range(
                0 + (gpu * heads_per_gpu),
                (0 + (gpu * heads_per_gpu)) + heads_per_gpu,
            )
        )
        for gpu in range(args.ngpu)
    }
    model.parallelize(device_map)
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for idx,subject in enumerate(subjects):
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain_mmlu]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = mmlu_eval(args, subject, model, tokenizer, dev_df, test_df, (idx,len(subjects)))
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("MMLU Average accuracy: {:.3f}".format(weighted_acc))
        
# BBH

def bbh_format_example(dataset, idx, include_answer=True):
    prompt = dataset["input"][idx]
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(dataset["target"][idx])
    return prompt


def bbh_gen_prompt(dataset , k=-1):
    prompt = ""
    if k == -1:
        k = len(dataset)
    for i in range(k):
        prompt += bbh_format_example(dataset, i)
    return prompt


def bbh_evaluate(model, dataset: Dataset, ntrain):
    data_train = dataset[:ntrain]
    data_test = dataset[ntrain:]
    is_correct = []

    for i in range(len(dataset) - ntrain):
        # get prompt and make sure it fits
        k = int(ntrain)
        prompt_end = bbh_format_example(data_test, i, include_answer=False)
        train_prompt = bbh_gen_prompt(data_train, k)
        prompt = train_prompt + prompt_end

        while not model.check_valid_length(prompt) and k > 0:
            k -= 1
            train_prompt = bbh_gen_prompt(data_train, k)
            prompt = train_prompt + prompt_end

        label = data_test["target"][i]
        pred = model.run(prompt)
        is_correct.append(pred.strip().startswith(label))

    return sum(is_correct) / len(is_correct)


def bbh_benchmark(model, ntrain = 3, data_dir = "lukaemon/bbh"):
    model.max_output_length = 32

    all_results = []
    data_names = get_dataset_config_names(data_dir)
    for idx, name in enumerate(data_names):
        dataset = load_dataset(data_dir, name, split="test")
        result = bbh_evaluate(model, dataset, ntrain=ntrain)
        all_results.append(result)
        print("Average accuracy {:.3f} - {}({}/{})".format(result, name, idx + 1, len(data_names)))

    score = (sum(all_results) / len(all_results))
    print("BBH Average accuracy: {:.3f}".format(score))

class EvalModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.max_input_length = self.model.seqlen
        self.max_output_length = 512

    def run(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=self.max_output_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def check_valid_length(self, text):
        inputs = self.tokenizer(text)
        return len(inputs.input_ids) <= self.max_input_length

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='t5 model to load'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--save_safetensors', type=str, default='',
        help='Save quantized `.safetensors` checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--benchmark', action='store_true',
        help='MMLU/BBH benchmarking'
    )
    parser.add_argument(
        '--benchmark_mode', default='mmlu' ,choices=['bbh','mmlu','both'],
        help='select benchmark dataset'
    )
    parser.add_argument(
        '--ntrain_mmlu', type=int, default=5,
        help='Number of k-shot to use for MMLU benchmarking.'
    )
    parser.add_argument(
        '--ntrain_bbh', type=int, default=3,
        help='Number of k-shot to use for BBH benchmarking.'
    )
    parser.add_argument(
        "--ngpu", "-g", type=int, default=1,
        help='Number of gpu to use for MMLU benchmarking.'
    )
    parser.add_argument(
        "--data_dir", "-d", type=str, default="data",
        help='MMLU dataset path'
    )
    
    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()
    
    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        model = get_t5(args.model)
        model.eval()
        
    if not args.nearest and not args.load:
        dataloader, testloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen * 2
        )

    if not args.load and args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = t5_sequential(model, dataloader, DEV)
        print(time.time() - tick)
    
    if not args.load and args.wbits < 16 and args.nearest:
        tick = time.time()
        quantizers = t5_nearest_sequential(model, DEV)
        print(time.time() - tick)
    
    if args.benchmark:
        model = model.to(DEV)
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(args.model)
        if args.benchmark_mode != 'bbh':
            mmlu_benchmark(model, tokenizer, args)
        
        if args.benchmark_mode != 'mmlu':
            evalmodel = EvalModel(model,tokenizer)
            bbh_benchmark(evalmodel, args.ntrain_bbh)

    if args.save:
        t5_pack(model, quantizers, args.wbits, args.groupsize)
        torch.save(model.state_dict(), args.save) 

    if args.save_safetensors:
        t5_pack(model, quantizers, args.wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save
        safe_save(model.state_dict(), args.save_safetensors)
