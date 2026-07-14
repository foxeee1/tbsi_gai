import sys
sys.path.insert(0, '.')
import torch.multiprocessing as mp
mp.set_start_method('forkserver', force=True)
print('Set forkserver start method')
