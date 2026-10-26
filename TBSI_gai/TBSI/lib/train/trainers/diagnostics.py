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
        self.amcm_grad_file = os.path.join(log_dir, 'amcm_gradient_flow.csv')
        if not os.path.exists(self.amcm_grad_file):
            with open(self.amcm_grad_file, 'w') as f:
                f.write('step,epoch,grad_norm,nonzero_params,total_params\n')

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

        # Feature-side residual statistics. These are sampled at the same
        # optimizer steps as gradient_flow.csv and make the intervention
        # attributable without changing the forward computation.
        self.feature_file = os.path.join(log_dir, 'feature_changes.csv')
        if not os.path.exists(self.feature_file):
            with open(self.feature_file, 'w') as f:
                f.write('step,epoch,module,rgb_in_norm,tir_in_norm,rgb_message_norm,'
                        'tir_message_norm,rgb_delta_norm,tir_delta_norm,rgb_out_norm,'
                        'tir_out_norm,gamma_rgb,gamma_tir,rgb_prob_mean,tir_prob_mean\n')

        self.stser_flow_file = os.path.join(log_dir, 'stser_flow.csv')
        if not os.path.exists(self.stser_flow_file):
            with open(self.stser_flow_file, 'w') as f:
                f.write('step,epoch,module,mode,competition,rgb_in_norm,tir_in_norm,'
                        'rgb_message_norm,tir_message_norm,rgb_delta_norm,tir_delta_norm,'
                        'rgb_relative_delta,tir_relative_delta,gamma_rgb,gamma_tir,'
                        'rgb_shared,rgb_private,rgb_null,tir_shared,tir_private,tir_null\n')
        self.stser_grad_file = os.path.join(log_dir, 'stser_gradient_flow.csv')
        if not os.path.exists(self.stser_grad_file):
            with open(self.stser_grad_file, 'w') as f:
                f.write('step,epoch,group,grad_norm,nonzero_tensors,total_tensors\n')
        self.content_enabled = any(
            module.__class__.__name__ == 'SearchConditionedT2SRouter' and
            getattr(module, 'mode', '') == 'stser_competition_content'
            for module in model.modules())
        if self.content_enabled:
            self.content_file = os.path.join(log_dir, 'stser_content_flow.csv')
            self.content_preclip_file = os.path.join(log_dir, 'stser_content_preclip_grad.csv')
            if not os.path.exists(self.content_file):
                with open(self.content_file, 'w') as f:
                    f.write('step,epoch,module,strength,modality,lambda,source_shift_l1,'
                            'foreground_mass_base,foreground_mass_changed,'
                            'valid_reference_fraction,content_message_norm,content_message_cosine_base,'
                            'route_shared,route_private,route_null\n')
            if not os.path.exists(self.content_preclip_file):
                with open(self.content_preclip_file, 'w') as f:
                    f.write('step,epoch,global_grad_norm,content_lambda_grad_norm,'
                            'content_lambda_has_grad\n')

        # TBSILayer hooks setup
        self._setup_tbsi_hooks(model)

    def _ensure_log_dir(self):
        # A restarted trainer or external cleanup must not make diagnostics
        # terminate an otherwise valid training step.
        os.makedirs(self.log_dir, exist_ok=True)

    def _setup_tbsi_hooks(self, model):
        """Register forward hooks on all TBSILayer modules to capture internal signals."""
        self._tbsi_signals = {}
        self._feature_changes = {}
        self._stser_flow = {}
        self._stser_content = {}
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'TBSILayer':
                layer_idx = name.split('.')[-1] if '.' in name else name
                module._diag_name = name
                module.register_forward_hook(self._make_tbsi_hook(name))
            if module.__class__.__name__ == 'PostTBSICrossModalResidual':
                module.register_forward_hook(self._make_residual_hook(name))
            if module.__class__.__name__ == 'AMCMLiteReliability':
                module.register_forward_hook(self._make_amcm_hook(name))
            if module.__class__.__name__ == 'SearchConditionedT2SRouter':
                module.register_forward_hook(self._make_stser_hook(name))

    def _make_stser_hook(self, name):
        def hook(module, inputs, outputs):
            try:
                _, _, rgb_in, tir_in = inputs
                rgb_out, tir_out = outputs
                stats = module.last_stats
                rgb_in_norm = rgb_in.detach().float().norm().item()
                tir_in_norm = tir_in.detach().float().norm().item()
                rgb_delta_norm = (rgb_out.detach().float() - rgb_in.detach().float()).norm().item()
                tir_delta_norm = (tir_out.detach().float() - tir_in.detach().float()).norm().item()
                self._feature_changes[name] = {
                    'rgb_in_norm': rgb_in_norm,
                    'tir_in_norm': tir_in_norm,
                    'rgb_message_norm': stats['rgb_message_norm'],
                    'tir_message_norm': stats['tir_message_norm'],
                    'rgb_delta_norm': rgb_delta_norm,
                    'tir_delta_norm': tir_delta_norm,
                    'rgb_out_norm': rgb_out.detach().float().norm().item(),
                    'tir_out_norm': tir_out.detach().float().norm().item(),
                    'gamma_rgb': stats['gamma_rgb'],
                    'gamma_tir': stats['gamma_tir'],
                }
                self._stser_flow[name] = {
                    'mode': stats['mode'],
                    'competition': int(stats['competition_mode']),
                    'rgb_in_norm': rgb_in_norm,
                    'tir_in_norm': tir_in_norm,
                    'rgb_message_norm': stats['rgb_message_norm'],
                    'tir_message_norm': stats['tir_message_norm'],
                    'rgb_delta_norm': rgb_delta_norm,
                    'tir_delta_norm': tir_delta_norm,
                    'rgb_relative_delta': rgb_delta_norm / (rgb_in_norm + 1e-8),
                    'tir_relative_delta': tir_delta_norm / (tir_in_norm + 1e-8),
                    'gamma_rgb': stats['gamma_rgb'],
                    'gamma_tir': stats['gamma_tir'],
                    'rgb_shared': stats.get('rgb_route_shared'),
                    'rgb_private': stats.get('rgb_route_private'),
                    'rgb_null': stats.get('rgb_route_null'),
                    'tir_shared': stats.get('tir_route_shared'),
                    'tir_private': stats.get('tir_route_private'),
                    'tir_null': stats.get('tir_route_null'),
                }
                if self.content_enabled and stats['mode'] == 'stser_competition_content':
                    self._stser_content[name] = dict(stats)
            except Exception:
                pass
        return hook

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

    def _make_residual_hook(self, name):
        def hook(module, input, output):
            try:
                rgb_in, tir_in = input
                rgb_out, tir_out = output
                rgb_message = module.rgb_from_tir(tir_in) if module.mode == 'diag' else None
                tir_message = module.tir_from_rgb(rgb_in) if module.mode == 'diag' else None
                self._feature_changes[name] = {
                    'rgb_in_norm': rgb_in.detach().float().norm().item(),
                    'tir_in_norm': tir_in.detach().float().norm().item(),
                    'rgb_message_norm': (rgb_message.detach().float().norm().item()
                                         if rgb_message is not None else 0.0),
                    'tir_message_norm': (tir_message.detach().float().norm().item()
                                         if tir_message is not None else 0.0),
                    'rgb_delta_norm': (rgb_out.detach().float() - rgb_in.detach().float()).norm().item(),
                    'tir_delta_norm': (tir_out.detach().float() - tir_in.detach().float()).norm().item(),
                    'rgb_out_norm': rgb_out.detach().float().norm().item(),
                    'tir_out_norm': tir_out.detach().float().norm().item(),
                    'gamma_rgb': float(module.gamma_rgb.detach().cpu()),
                    'gamma_tir': float(module.gamma_tir.detach().cpu()),
                }
            except Exception:
                pass
        return hook

    def _make_amcm_hook(self, name):
        def hook(module, input, output):
            try:
                rgb_in, tir_in = input[:2]
                rgb_out, tir_out, stats = output
                self._feature_changes[name] = {
                    'rgb_in_norm': rgb_in.detach().float().norm().item(),
                    'tir_in_norm': tir_in.detach().float().norm().item(),
                    'rgb_message_norm': stats.get('rgb_message_norm_live', torch.zeros(1)).detach().float().norm().item(),
                    'tir_message_norm': stats.get('tir_message_norm_live', torch.zeros(1)).detach().float().norm().item(),
                    'rgb_delta_norm': (rgb_out.detach().float() - rgb_in.detach().float()).norm().item(),
                    'tir_delta_norm': (tir_out.detach().float() - tir_in.detach().float()).norm().item(),
                    'rgb_out_norm': rgb_out.detach().float().norm().item(),
                    'tir_out_norm': tir_out.detach().float().norm().item(),
                    'gamma_rgb': 0.0,
                    'gamma_tir': 0.0,
                    'rgb_prob_mean': stats.get('rgb_prob_live', torch.zeros(1)).detach().float().mean().item(),
                    'tir_prob_mean': stats.get('tir_prob_live', torch.zeros(1)).detach().float().mean().item(),
                }
            except Exception:
                pass
        return hook

    def log_gradients(self, model, step, epoch):
        """Log gradient norms by module group: backbone vs head vs tbsi_layer."""
        self._ensure_log_dir()
        total_norm = 0.0
        backbone_norms = []
        head_norms = []
        tbsi_norms = []
        zero_grad = 0
        total_params = 0
        amcm_sq = 0.0
        amcm_nonzero = 0
        amcm_total = 0

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            total_params += 1
            if 'amcm_reliability' in name:
                amcm_total += 1
            if p.grad is None:
                zero_grad += 1
                continue

            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            if 'amcm_reliability' in name:
                amcm_sq += param_norm ** 2
                if param_norm > 0.0:
                    amcm_nonzero += 1

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
        with open(self.amcm_grad_file, 'a') as f:
            f.write(f'{step},{epoch},{amcm_sq ** 0.5:.8f},{amcm_nonzero},{amcm_total}\n')

        stser_groups = {'router': [0.0, 0, 0], 'experts': [0.0, 0, 0],
                        't2s_attention': [0.0, 0, 0], 'outer': [0.0, 0, 0]}
        for name, p in model.named_parameters():
            if 'search_conditioned_t2s' not in name:
                continue
            group = ('content' if 'content_logit' in name else
                     'router' if 'router_' in name else
                     'experts' if any(x in name for x in ('shared_expert', 'private_rgb', 'private_tir')) else
                     'outer' if name.endswith('gamma') or 'out_rgb' in name or 'out_tir' in name else
                     't2s_attention')
            if group not in stser_groups:
                stser_groups[group] = [0.0, 0, 0]
            values = stser_groups[group]
            values[2] += 1
            if p.grad is not None:
                norm = p.grad.data.norm(2).item()
                values[0] += norm * norm
                values[1] += int(norm > 0)
        with open(self.stser_grad_file, 'a') as f:
            for group, (sq_norm, nonzero, total) in stser_groups.items():
                f.write(f'{step},{epoch},{group},{sq_norm ** 0.5:.8f},{nonzero},{total}\n')

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
        self._ensure_log_dir()
        with open(self.loss_file, 'a') as f:
            f.write(f'{step},{epoch},'
                    f'{stats.get("Loss/total", 0):.6f},'
                    f'{stats.get("Loss/giou", 0):.6f},'
                    f'{stats.get("Loss/l1", 0):.6f},'
                    f'{stats.get("Loss/location", 0):.6f},'
                    f'{stats.get("IoU", 0):.6f}\n')

    def log_internal_signals(self, step, epoch):
        """Log TBSILayer internal feature norms from captured hooks."""
        self._ensure_log_dir()
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

    def log_feature_changes(self, step, epoch):
        self._ensure_log_dir()
        for name, values in self._feature_changes.items():
            with open(self.feature_file, 'a') as f:
                f.write(f'{step},{epoch},{name},'
                        f'{values["rgb_in_norm"]:.6f},{values["tir_in_norm"]:.6f},'
                        f'{values["rgb_message_norm"]:.6f},{values["tir_message_norm"]:.6f},'
                        f'{values["rgb_delta_norm"]:.6f},{values["tir_delta_norm"]:.6f},'
                    f'{values["rgb_out_norm"]:.6f},{values["tir_out_norm"]:.6f},'
                    f'{values["gamma_rgb"]:.8f},{values["gamma_tir"]:.8f},'
                    f'{values.get("rgb_prob_mean", 0.0):.8f},{values.get("tir_prob_mean", 0.0):.8f}\n')
        for name, values in self._stser_flow.items():
            with open(self.stser_flow_file, 'a') as f:
                f.write(f'{step},{epoch},{name},{values["mode"]},{values["competition"]},'
                        f'{values["rgb_in_norm"]:.6f},{values["tir_in_norm"]:.6f},'
                        f'{values["rgb_message_norm"]:.6f},{values["tir_message_norm"]:.6f},'
                        f'{values["rgb_delta_norm"]:.6f},{values["tir_delta_norm"]:.6f},'
                        f'{values["rgb_relative_delta"]:.8f},{values["tir_relative_delta"]:.8f},'
                        f'{values["gamma_rgb"]:.8f},{values["gamma_tir"]:.8f},'
                        f'{values["rgb_shared"]},{values["rgb_private"]},{values["rgb_null"]},'
                        f'{values["tir_shared"]},{values["tir_private"]},{values["tir_null"]}\n')
        if self.content_enabled:
            with open(self.content_file, 'a') as f:
                for name, stats in self._stser_content.items():
                    for modality in ('rgb', 'tir'):
                        prefix = modality + '_content_'
                        f.write(f'{step},{epoch},{name},{stats["content_strength"]:.6f},'
                                f'{modality},{stats.get(prefix + "lambda", 0.0):.8f},'
                                f'{stats.get(prefix + "source_shift_l1", 0.0):.8f},'
                                f'{stats.get(prefix + "foreground_mass_base", 0.0):.8f},'
                                f'{stats.get(prefix + "foreground_mass_changed", 0.0):.8f},'
                                f'{stats.get(prefix + "valid_reference_fraction", 0.0):.8f},'
                                f'{stats.get(prefix + "content_message_norm", 0.0):.8f},'
                                f'{stats.get(prefix + "content_message_cosine_base", 0.0):.8f},'
                                f'{stats.get(modality + "_route_shared", 0.0):.8f},'
                                f'{stats.get(modality + "_route_private", 0.0):.8f},'
                                f'{stats.get(modality + "_route_null", 0.0):.8f}\n')

    def log_stser_preclip(self, model, step, epoch):
        if not self.content_enabled or step % self.log_interval:
            return
        self._ensure_log_dir()
        all_sq, content_sq, has_grad = 0.0, 0.0, False
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            norm_sq = parameter.grad.detach().float().square().sum().item()
            all_sq += norm_sq
            if 'search_conditioned_t2s.content_logit' in name:
                content_sq += norm_sq
                has_grad = True
        with open(self.content_preclip_file, 'a') as f:
            f.write(f'{step},{epoch},{all_sq ** 0.5:.9g},'
                    f'{content_sq ** 0.5:.9g},{int(has_grad)}\n')

    def log_all(self, model, stats_dict, step, epoch):
        """Convenience: log all diagnostics in one call."""
        if step % self.log_interval != 0:
            return
        grad_info = self.log_gradients(model, step, epoch)
        self.log_loss_components(stats_dict, step, epoch)
        self.log_internal_signals(step, epoch)
        self.log_feature_changes(step, epoch)
        return grad_info
