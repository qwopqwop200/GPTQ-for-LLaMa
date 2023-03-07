import torch
import torch.nn as nn

import quant_cuda

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print('Benchmarking OPT-175B FC2 matvec ...')

DEV = torch.device('cuda:0')

M = 12288
N = 12288 * 4

DTYPE = torch.half
mat = torch.randn((M, N), device=DEV, dtype=DTYPE)
vec = torch.randn((1, M), device=DEV, dtype=DTYPE)
mul = torch.zeros((1, N), device=DEV, dtype=DTYPE)

COUNT = 1000
import time
tick = time.time()
for _ in range(COUNT):
    torch.matmul(vec, mat, out=mul) 
    torch.cuda.synchronize()
print('FP16:', (time.time() - tick) / COUNT)

DTYPE = torch.float
mat = mat.to(DTYPE)
vec = vec.to(DTYPE)
mul = mul.to(DTYPE)

mat = torch.randint(-1000000000, 1000000000, (M // 1024 * 128, N), device=DEV, dtype=torch.int)
scales = torch.randn(N, device=DEV, dtype=DTYPE)
zeros = torch.randn(N, device=DEV, dtype=DTYPE)

COUNT = 1000
import time
tick = time.time()
for _ in range(COUNT):
    quant_cuda.vecquant4matmul(vec, mat, mul, scales, zeros)
    torch.cuda.synchronize()
print('4bit:', (time.time() - tick) / COUNT)

print('Verifiying kernel correctness ...')

M = 4 * 4096
N = 4096

layer = nn.Linear(M, N)
vec = torch.randn(M).to(DEV)

from quant import *
quantizer = Quantizer()
quantizer.configure(4, perchannel=True, sym=False, mse=False)
quantizer.find_params(layer.weight.data, weight=True)
layer.weight.data = quantize(
    layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq
)

qlayer = Quant4Linear(layer.in_features, layer.out_features)
qlayer.pack(layer, quantizer.scale, quantizer.zero)

qlayer = qlayer.to(DEV)
layer = layer.to(DEV)

with torch.no_grad():
    print('Simu:', qlayer(vec))
    print('Kern:', layer.to(DEV)(vec))
