import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.models.layers.tbsi_layer import SignalAwareDecoupler


OUT_DIR = Path("output/diagnostics/signal_decouple_v162")
PARAMS = {
    "A": Path("output/experiments/v1.6.2-signal-decouple-residual-l12-A/checkpoints/TBSITrack_ep0015.pth.tar"),
    "B": Path("output/experiments/v1.6.2-signal-decouple-residual-l12-B/checkpoints/TBSITrack_ep0015.pth.tar"),
    "C": Path("output/experiments/v1.6.2-signal-decouple-residual-l12-C/checkpoints/TBSITrack_ep0015.pth.tar"),
}


def state_dict_from_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    for key in ["net", "model", "state_dict"]:
        if key in ckpt:
            return ckpt[key]
    return ckpt


def module_state(sd, prefix):
    out = {}
    stem = prefix + "."
    for key, value in sd.items():
        if key.startswith(stem):
            out[key[len(stem):]] = value
    return out


def grad_smoke(mod):
    torch.manual_seed(7)
    mod.train()
    x = torch.randn(2, 64, 768, requires_grad=True)
    signal, interference = mod(x)
    loss = (signal.pow(2).mean() + 0.1 * interference.pow(2).mean())
    loss.backward()
    grads = {}
    for name, p in mod.named_parameters():
        grads[name] = {
            "grad_abs_mean": float(p.grad.abs().mean().item()) if p.grad is not None else 0.0,
            "grad_abs_max": float(p.grad.abs().max().item()) if p.grad is not None else 0.0,
            "grad_is_none": p.grad is None,
        }
    with torch.no_grad():
        logits = mod.decouple(x.detach())
        modulation = mod.residual_scale * torch.tanh(logits)
    return grads, {
        "loss": float(loss.item()),
        "x_grad_abs_mean": float(x.grad.abs().mean().item()),
        "modulation_abs_mean": float(modulation.abs().mean().item()),
        "modulation_abs_max": float(modulation.abs().max().item()),
        "modulation_signed_mean": float(modulation.mean().item()),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    for seed, path in PARAMS.items():
        sd = state_dict_from_checkpoint(path)
        prefixes = sorted(
            {
                k.split(".decouple.")[0]
                for k in sd
                if ".signal_decoupler.decouple." in k and k.endswith(".weight")
            }
        )
        seed_report = {}
        for prefix in prefixes:
            msd = module_state(sd, prefix)
            mod = SignalAwareDecoupler(dim=768, mode="residual", residual_scale=0.5)
            mod.load_state_dict(msd, strict=True)
            stats = {}
            for name, tensor in msd.items():
                stats[name] = {
                    "abs_mean": float(tensor.abs().mean().item()),
                    "abs_max": float(tensor.abs().max().item()),
                    "l2": float(tensor.float().norm().item()),
                }
            grads, forward_stats = grad_smoke(mod)
            seed_report[prefix] = {
                "param_stats": stats,
                "grad_smoke": grads,
                "forward_smoke": forward_stats,
            }
        report[seed] = seed_report

    out_path = OUT_DIR / "checkpoint_modulation_gradient_stats.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"wrote": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
