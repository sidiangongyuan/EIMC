import time
import torch

class BlockTimeProfiler:
    def __init__(self, model, target_blocks):
        """
        Args:
            model: 要分析的模型
            target_blocks: 需要计时的模块名称列表 (对应model.named_modules里的名称)
        """
        self.model = model
        self.target_blocks = target_blocks
        self.time_records = {name: [] for name in target_blocks}
        self.hooks = []  # 用于保存hook句柄

    def _create_hook(self, name):
        def timing_hook(module, input, output):
            torch.cuda.synchronize()  # 确保CUDA操作同步
            start = time.perf_counter()  # 更高精度计时
            torch.cuda.synchronize()     # 确保前向计算完成
            elapsed = time.perf_counter() - start
            self.time_records[name].append(elapsed * 1000)  # 转换为毫秒
            
        return timing_hook

    def register_hooks(self):
        # 移除之前注册的所有hooks
        self.remove_hooks()
        
        # 只给目标模块注册hook
        for name, module in self.model.named_modules():
            if name in self.target_blocks:
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def profile(self, input_data, warmup=5, repeats=100):
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(input_data)
        
        # 正式测量
        self.register_hooks()
        try:
            for _ in range(repeats):
                _ = self.model(input_data)
        finally:
            self.remove_hooks()

        # 输出结果
        print("{:<20} | {:<10} | {:<10} | {:<10}".format(
            "Module", "Avg(ms)", "Min(ms)", "Max(ms)"))
        for name in self.target_blocks:
            times = self.time_records[name]
            if len(times) == 0:
                continue
            avg = sum(times) / len(times)
            min_t = min(times)
            max_t = max(times)
            print("{:<20} | {:<10.2f} | {:<10.2f} | {:<10.2f}".format(
                name, avg, min_t, max_t))