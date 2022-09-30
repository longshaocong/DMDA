def foo():
    batch = iter([1, 2, 3])
    while True:
        for b in batch:
            yield b 
            print('res:')
g = foo()
for i in range(20):
    print(next(g)) 