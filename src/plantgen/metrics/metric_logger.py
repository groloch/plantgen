from collections import deque, defaultdict

import mlflow


class Meter:
    def __init__(self, fmt='{avg:.4e}', window_size=100):
        self.fmt = fmt
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def update(self, value):
        self.values.append(value)

    def get_value(self):
        return sum(self.values) / len(self.values) if self.values else 0

    def __str__(self):
        avg = sum(self.values) / len(self.values) if self.values else 0
        return self.fmt.format(avg=avg)
        

class MetricLogger:
    def __init__(self, use_mlflow):
        self.metrics = defaultdict(Meter)
        self.use_mlflow = use_mlflow

    def log(self, step):
        names = self.metrics.keys()
        res = ''

        if len(names) == 0:
            return res

        for name in names:
            if self.use_mlflow:
                mlflow.log_metric(name, self.metrics[name].get_value(), step=step)
            if name.endswith('_avg'):
                res += f'{name[:-4]}: {self.metrics[name]}  '

        return res[:-2]
        
    def update(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = Meter(window_size=1)
        self.metrics[name].update(value)
        self.metrics[f'{name}_avg'].update(value)
