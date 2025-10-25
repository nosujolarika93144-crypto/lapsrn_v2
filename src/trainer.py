import os
import math
from decimal import Decimal
import utility
from utility import calc_gradient_metrics 
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model.model)

        # [解决方案] 在这里加载优化器状态，并使用 my_loss.log 的长度来获取正确的历史周期数
        if self.args.load != '':
            try:
                num_epochs_trained = len(self.loss.log)
                self.optimizer.load(ckp.dir, epoch=num_epochs_trained)
            except Exception as e:
                # [解决方案] 捕获所有可能的加载错误，包括 FileNotFoundError 和 EOFError
                print(f"[警告] 加载优化器状态失败: {e}")
                print("[警告] 将使用一个全新的优化器从周期 {num_epochs_trained + 1} 开始。")
                # 即使优化器加载失败，我们仍然需要快进学习率调度器
                if num_epochs_trained > 0:
                    print(f"[信息] 正在将学习率调度器快进到周期 {num_epochs_trained}。")
                    # 这个循环可以确保调度器处于正确的状态
                    self.optimizer.scheduler.last_epoch = -1
                    for _ in range(num_epochs_trained):
                        self.optimizer.scheduler.step()
                print("[警告] 找不到旧的优化器状态文件(optimizer.pt)，将从头开始训练优化器。")

        self.error_last = 1e8
        self.best_metric = float('inf')

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
        self.ckp.write_log(f'[Epoch {epoch}]\tLearning rate: {Decimal(lr):.2e}')
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        
        for batch, (lrs, hr, _,) in enumerate(self.loader_train):
            lrs = self.prepare(*lrs)
            hr, = self.prepare(hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            srs = self.model(*lrs)
            
            hr_x4 = F.interpolate(hr, scale_factor=1/4, mode='bicubic', align_corners=False)
            hr_x2 = F.interpolate(hr, scale_factor=1/2, mode='bicubic', align_corners=False)
            hrs = [hr_x4, hr_x2, hr]
            
            loss_weights = [0.2, 0.3, 0.5]

            total_loss = 0
            for i, (sr_intermediate, hr_intermediate) in enumerate(zip(srs, hrs)):
                stage_loss = self.loss(sr_intermediate, hr_intermediate)
                weighted_stage_loss = loss_weights[i] * stage_loss
                total_loss += weighted_stage_loss
            # ============= [ 新增代码：熔断机制 ] =============
            # 检查损失值是否有效，防止梯度爆炸导致的崩溃
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"[警告] 在 Epoch {epoch}, Batch {batch} 中检测到无效的损失值 (NaN or Inf)。跳过此批次的更新。")
                self.optimizer.zero_grad() # 清除可能存在的坏梯度
                continue # 直接进入下一个批次
            # =================================================
                      
            total_loss.backward()

            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.model.parameters(),
                    self.args.gclip
                )
                
            self.optimizer.step()
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log(f'[{((batch + 1) * self.args.batch_size)}/{len(self.loader_train.dataset)}]\t'
                                   f'{self.loss.display_loss(batch)}\t{timer_model.release():.1f}+{timer_data.release():.1f}s')
            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        timer_test = utility.timer()
        
        # [解决方案] 将 'current_epoch_metric' 的计算移到每个数据集循环的末尾
        # self.best_metric 仍然用于跟踪最佳模型的性能
        
        for idx_data, d in enumerate(self.loader_test):
            # [解决方案] 为每个输出尺度创建一个独立的指标字典
            # 您的模型输出3个尺度 (x4, x2, x1), 所以我们创建3个字典
            total_metrics = [
                {'rel_l2_error': 0.0, 'vorticity_error': 0.0, 'mean_abs_divergence': 0.0}
                for _ in range(len(self.scale)) # 长度为3
            ]
            
            for lrs, hr, filename in tqdm(d, ncols=80):
                lrs = self.prepare(*lrs)
                hr_norm, = self.prepare(hr)
                
                # 模型输出一个包含3个尺度SR结果的列表
                srs_norm = self.model(*lrs)

                # [解决方案] 创建与训练时逻辑一致的多尺度真实值(ground truth)
                # srs_norm[0] -> x4, srs_norm[1] -> x2, srs_norm[2] -> x1
                hrs_norm = [
                    F.interpolate(hr_norm, scale_factor=0.25, mode='bicubic', align_corners=False), # 对应 img_out_x4
                    F.interpolate(hr_norm, scale_factor=0.5, mode='bicubic', align_corners=False),  # 对应 img_out_x2
                    hr_norm                                          # 对应 img_out_x1
                ]
                
                # [解决方案] 循环遍历每个尺度，分别计算和累积指标
                for i, (sr_norm, hr_intermediate_norm) in enumerate(zip(srs_norm, hrs_norm)):
                    # [解决方案] 从args读取统计值
                    mean = torch.tensor([self.args.u_mean, self.args.v_mean], device=sr_norm.device).view(1, 2, 1, 1)
                    std = torch.tensor([self.args.u_std, self.args.v_std], device=sr_norm.device).view(1, 2, 1, 1)
                    
                    sr_unnormalized = sr_norm * std + mean
                    hr_unnormalized = hr_intermediate_norm * std + mean
                    
                    sr_unnormalized_numpy = sr_unnormalized.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    hr_unnormalized_numpy = hr_unnormalized.squeeze(0).permute(1, 2, 0).cpu().numpy()

                    l2_diff_norm = np.linalg.norm(sr_unnormalized_numpy - hr_unnormalized_numpy)
                    l2_hr_norm = np.linalg.norm(hr_unnormalized_numpy)
                    if l2_hr_norm > 1e-10:
                        total_metrics[i]['rel_l2_error'] += l2_diff_norm / l2_hr_norm
                    
                    sr_grad_metrics = calc_gradient_metrics(sr_unnormalized_numpy)
                    hr_grad_metrics = calc_gradient_metrics(hr_unnormalized_numpy)
                    
                    vorticity_diff_norm = np.linalg.norm(sr_grad_metrics['vorticity_field'] - hr_grad_metrics['vorticity_field'])
                    vorticity_hr_norm = np.linalg.norm(hr_grad_metrics['vorticity_field'])
                    
                    if vorticity_hr_norm > 1e-10:
                        total_metrics[i]['vorticity_error'] += vorticity_diff_norm / vorticity_hr_norm
                    
                    total_metrics[i]['mean_abs_divergence'] += sr_grad_metrics['mean_abs_divergence']

            # [解决方案] 为每个尺度打印评估结果
            # 注意：这里的 self.scale 是 [8, 4, 2]，而模型输出是 x4, x2, x1 上采样结果
            output_scales = ['x4', 'x2', 'x1 (Full Resolution)']
            for i, metrics in enumerate(total_metrics):
                log_str = '[{} {}]'.format(d.dataset.name, output_scales[i])
                for key, value in metrics.items():
                    avg_metric = value / len(d) if len(d) > 0 else 0
                    log_str += f'\t{key}: {avg_metric:.6f}'
                self.ckp.write_log(log_str)
            
            # [解决方案] 仍然使用最终分辨率(full resolution)的 rel_l2_error 作为判断最佳模型的标准
            current_epoch_metric = total_metrics[-1]['rel_l2_error'] / len(d) if len(d) > 0 else float('inf')

            is_best = current_epoch_metric < self.best_metric
            if is_best:
                self.best_metric = current_epoch_metric
                self.ckp.write_log(f'[INFO] New best model found with rel_l2_error on final output: {self.best_metric:.6f}')

        self.ckp.write_log(f'Forward: {timer_test.toc():.2f}s\n')
        self.ckp.write_log('Saving...')

        if self.args.save_results: self.ckp.end_background()
        if not self.args.test_only: self.ckp.save(self, epoch, is_best=is_best)

        self.ckp.write_log(f'Total: {timer_test.toc():.2f}s\n', refresh=True)
        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs