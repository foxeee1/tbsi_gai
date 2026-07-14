"""
Lightweight training diagnostics for TBSI experiments.

Provides three monitoring capabilities with < 5% overhead:
1. Gradient flow monitoring (per-module-group grad norms)
2. Loss component decomposition
3. Internal signal capture (TBSILayer cross-attention patterns)

Usage:
    from .diagnostics import Diagnostics

    diag = Diagnostics(model, log_dir='./output/logs')
    # ... in training loop ...
    diag.log_gradients(model, step, epoch)
    diag.log_loss_components(loss_dict, step, epoch)
    diag.log_internal_signals(model, step, epoch)
"""
import os
import numpy as np
import torch


class Diagnostics:
    def __init__(self, model, log_dir, log_interval=50):
        self.log_interval = log_interval
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Gradient flow CSV
        self.grad_file = os.path.join(log_dir, 'gradient_flow.csv')
        if not os.path.exists(self.grad_file):
            with open(self.grad_file, 'w') as f:
                f.write('step,epoch,total_norm,backbone_mean,head_mean,zero_grad_count,total_params\n')

        # Loss components CSV
        self.loss_file = os.path.join(log_dir, 'loss_components.csv')
        if not os.path.exists(self.loss_file):
            with open(self.loss_file, 'w') as f:
                f.write('step,epoch,total,giou,l1,location,iou\n')

        # Internal signals CSV (TBSILayer cross-attention patterns)
        self.signal_file = os.path.join(log_dir, 'internal_signals.csv')
        if not os.path.exists(self.signal_file):
            with open(self.signal_file, 'w') as f:
                f.write('step,epoch,tbsi_layer,fused_t_norm,temp_xv_norm,temp_xi_norm,'
                        'xv_template_norm,xi_template_norm,xv_search_norm,xi_search_norm\n')

        # TBSILayer hooks setup
        self._setup_tbsi_hooks(model)

    def _setup_tbsi_hooks(self, model):
        """Register forward hooks on all TBSILayer modules to capture internal signals."""
        self._tbsi_signals = {}
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'TBSILayer':
                layer_idx = name.split('.')[-1] if '.' in name else name
                module._diag_name = name
                module.register_forward_hook(self._make_tbsi_hook(name))

    def _make_tbsi_hook(self, name):
        """Create a forward hook that captures TBSILayer internal feature norms."""
        def hook(module, input, output):
            try:
                x_v_out, x_i_out = output
                # x_v: [B, N, C], x_i: [B, N, C] where N = 64 template + 256 search
                if x_v_out.ndim == 3:
                    B, N, C = x_v_out.shape
                    lens_z = N - 256 if N > 256 else 64  # heuristic: N - 256 = template tokens

                    self._tbsi_signals[name] = {
                        'xv_template_norm': x_v_out[:, :lens_z, :].norm().item(),
                        'xv_search_norm': x_v_out[:, lens_z:, :].norm().item(),
                        'xi_template_norm': x_i_out[:, :lens_z, :].norm().item(),
                        'xi_search_norm': x_i_out[:, lens_z:, :].norm().item(),
                        'ratio_vt_vs': (x_v_out[:, :lens_z, :].norm() / (x_v_out[:, lens_z:, :].norm() + 1e-8)).item(),
                        'ratio_it_is': (x_i_out[:, :lens_z, :].norm() / (x_i_out[:, lens_z:, :].norm() + 1e-8)).item(),
                    }
            except Exception:
                pass
        return hook

    def log_gradients(self, model, step, epoch):
        """Log gradient norms by module group: backbone vs head vs tbsi_layer."""
        total_norm = 0.0
        backbone_norms = []
        head_norms = []
        tbsi_norms = []
        zero_grad = 0
        total_params = 0

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            total_params += 1
            if p.grad is None:
                zero_grad += 1
                continue

            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2

            if 'backbone' in name:
                backbone_norms.append(param_norm)
            elif 'head' in name or 'box_head' in name or 'score_head' in name:
                head_norms.append(param_norm)
            elif 'ca_' in name and 'tbsi' in name.lower():
                tbsi_norms.append(param_norm)

        total_norm = total_norm ** 0.5
        b_mean = np.mean(backbone_norms) if backbone_norms else -1.0
        h_mean = np.mean(head_norms) if head_norms else -1.0
        t_mean = np.mean(tbsi_norms) if tbsi_norms else -1.0

        with open(self.grad_file, 'a') as f:
            f.write(f'{step},{epoch},{total_norm:.6f},{b_mean:.6f},{h_mean:.6f},'
                    f'{zero_grad},{total_params}\n')

        # Alert on anomalies (skip exact-zero — sampled post-zero_grad)
        if 0 < total_norm < 1e-7:
            print(f'  ⚠️ [Step {step}] Total grad norm near zero! Gradient vanishing.')
        elif total_norm > 100:
            print(f'  🔥 [Step {step}] Total grad norm={total_norm:.2f}, gradient explosion!')
        if 0 < zero_grad < total_params * 0.3:
            print(f'  💀 [Step {step}] {zero_grad}/{total_params} params have zero gradient.')

        return {'total_grad_norm': total_norm, 'backbone_grad_mean': b_mean,
                'head_grad_mean': h_mean, 'tbsi_grad_mean': t_mean}

    def log_loss_components(self, stats, step, epoch):
        """Log individual loss components (already computed in actor)."""
        with open(self.loss_file, 'a') as f:
            f.write(f'{step},{epoch},'
                    f'{stats.get("Loss/total", 0):.6f},'
                    f'{stats.get("Loss/giou", 0):.6f},'
                    f'{stats.get("Loss/l1", 0):.6f},'
                    f'{stats.get("Loss/location", 0):.6f},'
                    f'{stats.get("IoU", 0):.6f}\n')

    def log_internal_signals(self, step, epoch):
        """Log TBSILayer internal feature norms from captured hooks."""
        for name, signals in self._tbsi_signals.items():
            layer_short = name.replace('backbone.', '').replace('blocks.', 'b').replace('.tbsi_layer', '_tbsi')
            with open(self.signal_file, 'a') as f:
                f.write(f'{step},{epoch},{layer_short},'
                        f'{signals.get("ratio_vt_vs", 0):.4f},'
                        f'{signals.get("ratio_it_is", 0):.4f},'
                        f'{signals.get("xv_template_norm", 0):.4f},'
                        f'{signals.get("xi_template_norm", 0):.4f},'
                        f'{signals.get("xv_search_norm", 0):.4f},'
                        f'{signals.get("xi_search_norm", 0):.4f}\n')

    def log_all(self, model, stats_dict, step, epoch):
        """Convenience: log all diagnostics in one call."""
        if step % self.log_interval != 0:
            return
        grad_info = self.log_gradients(model, step, epoch)
        self.log_loss_components(stats_dict, step, epoch)
        self.log_internal_signals(step, epoch)
        return grad_info
