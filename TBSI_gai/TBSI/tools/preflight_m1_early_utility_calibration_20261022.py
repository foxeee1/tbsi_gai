import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lib.config.tbsi_track.config import cfg as default_cfg
from lib.config.tbsi_track.config import update_config_from_file
from lib.models.tbsi_track.tbsi_track import build_tbsi_track
from lib.train.admin.settings import Settings
from lib.train.actors.tbsi_track import TBSITrackActor
from lib.train.base_functions import build_dataloaders, update_settings
from lib.utils.box_ops import giou_loss
from lib.utils.focal_loss import FocalLoss
from torch.nn.functional import l1_loss


def load_cfg(path):
    config = copy.deepcopy(default_cfg)
    update_config_from_file(str(path), config)
    config.TRAIN.NUM_WORKER = 0
    config.TRAIN.PREFETCH_FACTOR = None
    config.TRAIN.PERSISTENT_WORKERS = False
    return config


def move_batch(batch, settings, epoch=1):
    if isinstance(batch, dict):
        for key in ('visible', 'infrared'):
            if batch.get(key):
                batch[key] = batch[key].to('cuda', non_blocking=True)
    batch['epoch'] = epoch
    batch['settings'] = settings
    return batch


def setup_loader(config, sequence_file, samples, settings):
    os.environ['TBSI_RPP_TRAIN_LIST'] = str(sequence_file)
    config.DATA.TRAIN.DATASETS_NAME = ['RPP_LasHeR_train']
    config.DATA.TRAIN.DATASETS_RATIO = [1]
    config.DATA.TRAIN.SAMPLE_PER_EPOCH = int(samples)
    config.DATA.VAL.DATASETS_NAME = ['LasHeR_test']
    config.DATA.VAL.DATASETS_RATIO = [1]
    config.DATA.VAL.SAMPLE_PER_EPOCH = 16
    return build_dataloaders(config, settings)[0]


def utility_targets(prediction, visible):
    # Reuse the actor's exact target-map generation and threshold for parity.
    from lib.utils.heapmap_utils import generate_heatmap
    heat = generate_heatmap(visible['search_anno'], 256, 16)[-1].unsqueeze(1)
    positive = heat >= 0.5
    rgb, tir = prediction['early_utility_teacher_score_maps']
    def margin(score):
        flat, mask = score.flatten(1), positive.flatten(1)
        fg = flat.masked_fill(~mask, float('-inf')).max(dim=1).values
        bg = flat.masked_fill(mask, float('-inf')).max(dim=1).values
        return fg - bg
    utilities = torch.stack((margin(rgb), margin(tir)), dim=1).detach()
    labels = utilities.argmax(dim=1)
    regrets = utilities.max(dim=1).values - utilities.gather(1, prediction[
        'early_utility_logits'].argmax(dim=1, keepdim=True)).squeeze(1)
    return utilities, labels, regrets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--quick-sequences', required=True)
    parser.add_argument('--calibration-sequences', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fit-updates', type=int, default=32)
    parser.add_argument('--calibration-samples', type=int, default=768)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this utility calibration')

    seed = 42022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    config = load_cfg(args.config)
    settings = Settings()
    settings.local_rank = -1
    settings.use_lmdb = False
    update_settings(settings, config)
    settings.batchsize = config.TRAIN.BATCH_SIZE

    fit_loader = setup_loader(config, args.quick_sequences,
                              args.fit_updates * config.TRAIN.BATCH_SIZE, settings)
    model = build_tbsi_track(config, training=True).cuda()
    model.train()
    objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': FocalLoss()}
    actor = TBSITrackActor(
        net=model, objective=objective,
        loss_weight={'giou': config.TRAIN.GIOU_WEIGHT,
                     'l1': config.TRAIN.L1_WEIGHT, 'focal': 1.0, 'cls': 1.0},
        settings=settings, cfg=config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.TRAIN.LR,
                                  weight_decay=config.TRAIN.WEIGHT_DECAY)
    fit_labels = []
    fit_losses = []
    for step, batch in enumerate(fit_loader):
        batch = move_batch(batch, settings)
        optimizer.zero_grad(set_to_none=True)
        prediction = actor.forward_pass(batch)
        _, labels, _ = utility_targets(prediction, batch['visible'])
        fit_labels.append(labels.detach().cpu())
        loss, _ = actor.compute_losses(prediction, batch['visible'])
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite temporary fit loss at step {}'.format(step + 1))
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in trainable):
            raise RuntimeError('non-finite router gradient at step {}'.format(step + 1))
        optimizer.step()
        fit_losses.append(float(loss.detach()))
        if step + 1 >= args.fit_updates:
            break
    if len(fit_losses) != args.fit_updates:
        raise RuntimeError('temporary fit ended before requested update count')
    del fit_loader, optimizer

    calibration_loader = setup_loader(config, args.calibration_sequences,
                                      args.calibration_samples, settings)
    model.eval()
    all_logits, all_labels, all_utilities, all_regrets = [], [], [], []
    with torch.no_grad():
        for step, batch in enumerate(calibration_loader):
            batch = move_batch(batch, settings)
            pred = actor.forward_pass(batch)
            if 'early_utility_teacher_score_maps' not in pred:
                raise RuntimeError('calibration batch has no counterfactual teacher outputs')
            utilities, labels, regrets = utility_targets(pred, batch['visible'])
            all_logits.append(pred['early_utility_logits'].float().cpu())
            all_labels.append(labels.cpu())
            all_utilities.append(utilities.cpu())
            all_regrets.append(regrets.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    utilities = torch.cat(all_utilities)
    regrets = torch.cat(all_regrets)
    fit_counts = torch.bincount(torch.cat(fit_labels), minlength=2).float()
    prior = (fit_counts + 1.0) / (fit_counts.sum() + 2.0)
    router_ce = float(F.cross_entropy(logits, labels))
    prior_ce = float((-prior[labels].log()).mean())
    random_regret = float((utilities[:, 0] - utilities[:, 1]).abs().mean() * 0.5)
    mean_regret = float(regrets.mean())
    result = {
        'status': 'CALIBRATION-GATE-PASSED' if router_ce < prior_ce and mean_regret < random_regret
                  else 'CALIBRATION-GATE-FAILED',
        'seed': seed,
        'temporary_fit_updates': len(fit_losses),
        'temporary_fit_loss_first': fit_losses[0],
        'temporary_fit_loss_last': fit_losses[-1],
        'calibration_samples': int(labels.numel()),
        'fit_label_counts_rgb_tir': fit_counts.tolist(),
        'calibration_label_counts_rgb_tir': torch.bincount(labels, minlength=2).tolist(),
        'router_utility_cross_entropy': router_ce,
        'constant_prior_cross_entropy': prior_ce,
        'router_mean_utility_regret': mean_regret,
        'random_modality_expected_utility_regret': random_regret,
        'predicted_modality_accuracy': float((logits.argmax(1) == labels).float().mean()),
        'input_cohorts': {
            'quick_train': str(Path(args.quick_sequences).resolve()),
            'quick_train_sha256': __import__('hashlib').sha256(
                Path(args.quick_sequences).read_bytes()).hexdigest(),
            'calibration': str(Path(args.calibration_sequences).resolve()),
            'calibration_sha256': __import__('hashlib').sha256(
                Path(args.calibration_sequences).read_bytes()).hexdigest(),
        },
        'interpretation': 'Sequence-held-out random crop utility prediction; not a fixed-state replay.',
        'temporary_weights_persisted': False,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
