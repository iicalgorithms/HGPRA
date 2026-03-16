import numpy as np
from sklearn.metrics import f1_score, accuracy_score


EOS = 1e-10


class Eval_Metrics:
    def __init__(self, y_true, y_pred_logit):
        y_pred = np.argmax(y_pred_logit, axis=1)
        
        self.acc = accuracy_score(y_true, y_pred)
        self.macro_f1 = f1_score(y_true, y_pred, average='macro')
        self.micro_f1 = f1_score(y_true, y_pred, average='micro')
    
    def __lt__(self, other):
        return self.acc < other.acc

    def __str__(self):
        return f'acc: {self.acc:.4f}, macro F1: {self.macro_f1:.4f}, micro F1: {self.micro_f1:.4f}'


class Eval_Metrics_Average:
    def __init__(self):
        self.eval_metrics_list = []

    def get_metric(self, metric):
        accs = [x.acc for x in self.eval_metrics_list]
        macro_f1s = [x.macro_f1 for x in self.eval_metrics_list]
        micro_f1s = [x.micro_f1 for x in self.eval_metrics_list]
        assert metric in ['acc', 'macro_f1', 'micro_f1']
        if metric == 'acc':
            return self.__get_average(accs)
        if metric == 'macro_f1':
            return self.__get_average(macro_f1s)
        if metric == 'micro_f1':
            return self.__get_average(micro_f1s)

    def __get_average(self, num_list):
        return np.mean(num_list)

    def __get_std(self, num_list):
        return np.std(num_list)

    def __call__(self, eval_metrics):
        self.eval_metrics_list.append(eval_metrics)
    
    def __str__(self):
        accs = [x.acc for x in self.eval_metrics_list]
        macro_f1s = [x.macro_f1 for x in self.eval_metrics_list]
        micro_f1s = [x.micro_f1 for x in self.eval_metrics_list]
        
        return f'Average of {len(self.eval_metrics_list)} runs:' \
               f'Acc: {self.__get_average(accs):.4f} ± {self.__get_std(accs):.4f}, ' \
               f'macro F1: {self.__get_average(macro_f1s):.4f} ± {self.__get_std(macro_f1s):.4f}, ' \
               f'micro F1: {self.__get_average(micro_f1s):.4f} ± {self.__get_std(micro_f1s):.4f}, '
