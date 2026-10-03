"""观测模型前向耗时，不替换前向方法、模型权重或生成算法。"""

import time


class ForwardTiming:
    def __init__(self, torch, device):
        self.torch = torch
        self.cuda = str(device).startswith('cuda')
        self.measurements = []

    def before(self, module=None, args=None):
        if self.cuda:
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start, end = time.perf_counter(), None
        self.measurements.append([start, end])

    def after(self, module=None, args=None, result=None):
        if self.cuda:
            self.measurements[-1][1].record()
        else:
            self.measurements[-1][1] = time.perf_counter()

    def milliseconds(self):
        if self.cuda and self.measurements:
            self.measurements[-1][1].synchronize()
        return [float(start.elapsed_time(end)) if self.cuda else (end - start) * 1000
                for start, end in self.measurements]
