import torch
import torch.nn as nn
import math
import numpy as np

class ECGLoss:
    def __init__(self, class_dict):
        self.class_dict = class_dict
    
    def cal_weights(self):
        total = sum(self.class_dict.values())
        max_num = max(self.class_dict.values())
        mu = 1.0 / (total / max_num)
        class_weight = dict()
        for key, value in self.class_dict.items():
            score = math.sqrt(mu * total / float(value))
            class_weight[key] = score if score > 1.0 else 1.0
        return class_weight
    
    def criterion(self):
        class_weight = self.cal_weights()
        weights_array = np.array(list(class_weight.values()), dtype=np.float32)
        weights_tensor = torch.tensor(weights_array)
        loss_function = nn.CrossEntropyLoss(weight=weights_tensor)
        return loss_function
