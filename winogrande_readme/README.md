# Winogrande Tests for LLaMa

## Scores from the Modelcard
Facebook published results for the Winogrande test dataset in the LLaMa modelcard:

https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md

| Number of parameters |WinoGrande score|
| --- |  --------- |
| 7B   |  70.1     |
| 13B  |  73       |
| 33B |   76       |
| 65B |   77       |

The parameters or prompt that was used for the test is currently unknown. 
(see https://github.com/facebookresearch/llama/issues/188)

##Test script

In order to test, validate  and compare the quantization I wrote small script that 
fetches the dataset from huggingface datasets and asks the model for the correct answer.

A test prompt is hardcoded in the script.  

The script can be run with:

```
python llama_winogrande.py decapoda-research/llama-7b-hf --wbits 4 --load llama-7b-4bit.pt
```
## Results

At the moment I have only run it with the 7B 4bit GPTQ version. But the results are bad. (48% correct in a multiple choice test with two options)

It is possible that the prompt or parameters need to be changed.  