import multiprocessing


def _temp_run(shared):
    d = {}
    for i in range(int(1e1)):
        d[str(i)] = i
    shared.append(103)
    return 103

manager = multiprocessing.Manager()
# result = []
result = manager.list()
p = multiprocessing.Process(target=_temp_run, args=(result,))
p.start()
# p.join(timeout=10)
p.join()
if p.is_alive():
    print('kill p')
    p.kill()
print(result)

result2 = manager.list()
print(result2)
print("didn't fail")

print(str(result))
print(repr(result))
