import matplotlib.pyplot as plt
from numpy import size
import torch
import global_consts
from Backend.NV_Multiprocessing.RenderProcess import SparseCache as Cache

cache = Cache((10, 20), torch.device("cpu"))
data = torch.tile(torch.unsqueeze(torch.linspace(0, 19, 20), 0), (3, 1))
cache.write(data, 7, 10)
print(cache.read(7, 9))
print(cache.read(2, 5))
print(cache.read(5, 8))
cache.write(data * 2, 5, 8)
print(cache.read(0, 10))