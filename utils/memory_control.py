import gc
import psutil
import os

process = psutil.Process(os.getpid())


def mem():
    print('Memory usage, physical memory: %d' % process.memory_info().rss)
    print('Memory usage, virtual memory: %d' % process.memory_info().vms)
    print('Memory usage, physical memory percentage: %d' % process.memory_percent(memtype="rss"))
    print('GC collected objects : %d' % gc.collect())
