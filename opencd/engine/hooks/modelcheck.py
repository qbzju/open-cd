from mmengine.hooks import Hook
import torch
from mmengine.logging import MMLogger
from opencd.registry import HOOKS

@HOOKS.register_module()  # 添加注册器
class ModelCheckHook(Hook):
    def __init__(self, check_frequency=100, **kwargs):  # 添加 **kwargs 接收额外参数
        super().__init__()  # 调用父类的初始化
        self.check_frequency = check_frequency
        self.logger = MMLogger.get_current_instance()
        
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """在每次训练迭代后调用"""
        if batch_idx % self.check_frequency == 0:
            model = runner.model
            
            # 检查权重
            for name, param in model.named_parameters():
                if param.requires_grad:  # 只检查需要梯度的参数
                    stats = {
                        'mean': param.data.mean().item(),
                        'std': param.data.std().item(),
                        'max': param.data.max().item(),
                        'min': param.data.min().item()
                    }
                    
                    # 检查数值是否异常
                    if torch.isnan(param.data).any():
                        self.logger.error(f"NaN found in {name}")
                        import pdb; pdb.set_trace()
                    if torch.isinf(param.data).any():
                        self.logger.error(f"Inf found in {name}")
                        import pdb; pdb.set_trace()
                        
                    # 检查梯度
                    if param.grad is not None:
                        grad_stats = {
                            'mean': param.grad.mean().item(),
                            'std': param.grad.std().item(),
                            'max': param.grad.max().item(),
                            'min': param.grad.min().item()
                        }
                        if torch.isnan(param.grad).any():
                            self.logger.error(f"NaN gradient found in {name}")
                            import pdb; pdb.set_trace()
                        if torch.isinf(param.grad).any():
                            self.logger.error(f"Inf gradient found in {name}")
                            import pdb; pdb.set_trace()
                            
                        self.logger.info(f"Layer: {name}\nWeight stats: {stats}\nGrad stats: {grad_stats}\n")