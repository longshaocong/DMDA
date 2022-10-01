# def foo():
#     batch = iter([1, 2, 3])
#     while True:
#         for b in batch:
#             yield b 
#             print('res:')
# g = foo()
# for i in range(20):
#     print(next(g)) 
import torch
class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


x = [0, 1, 2, 3, 4]
sampler = torch.utils.data.RandomSampler(x, replacement=False)
for i in sampler:
    print(i)
print('-----------------')
for i in sampler:
    print(i)
batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=2, drop_last=False)
for i in batch_sampler:
    print(i)
print('===================')
for i in batch_sampler:
    print(i)
print('===================')
new_batch_sampler = iter(_InfiniteSampler(batch_sampler))
for i in range(20):
    print(next(new_batch_sampler))


