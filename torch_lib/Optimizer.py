import torch
import torch.nn as nn


#optimizer for vgg model
class NL_Optimizer(torch.optim.Optimizer):
    def __init__(self, configs):
        """
        input
        [
            {'params': ..., 'type': torch.optim.AdamW, 'lr': 1e-3, 'period': 1},
            {'params': ..., 'type': torch.optim.SGD,   'lr': 1e-4, 'period': 10, 'momentum': 0.9}
        ]
        """
        self.optimizers = []
        self.step_cnt = 0
        all_groups = []
        for config in configs:
            period = config.pop('period', 1)
            opt_type = config.pop('type', torch.optim.AdamW) 
            params = config.pop('params')
            
            opt = opt_type(params, **config)
            all_groups.extend(opt.param_groups)
            self.optimizers.append({
                'opt': opt,
                'period': period,
                'type': opt_type.__name__ 
            })
        super(NL_Optimizer, self).__init__(all_groups, defaults={})

    @torch.no_grad()    #override step
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_cnt += 1
        
        for item in self.optimizers:
            period = item['period']
            opt = item['opt']
            if self.step_cnt % period == 0:
                opt.step()
                opt.zero_grad() 

        return loss


    def zero_grad(self):
        """dummy funtion to prevent zero out the gradients accidientally"""
        pass 

    
    def state_dict(self):
        return {
            'step_cnt': self.step_cnt,
            'opts': [item['opt'].state_dict() for item in self.optimizers]
        }

    def load_state_dict(self, state_dict):
        self.step_cnt = state_dict.get('step_cnt', 0)
        saved_opts = state_dict.get('opts', [])
        
        if len(saved_opts) != len(self.optimizers):
            print("Warning: Loaded optimizer state count mismatch!")
        
        for i, saved_state in enumerate(saved_opts):
            if i < len(self.optimizers):
                self.optimizers[i]['opt'].load_state_dict(saved_state)
                
                