import argparse
import datetime as dt
import os
import signal
import subprocess
import time
from pathlib import Path


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def newest_checkpoint(ckpt_dir: Path) -> str:
    if not ckpt_dir.exists():
        return "missing"
    ckpts = sorted(ckpt_dir.glob("TBSITrack_ep*.pth.tar"))
    if not ckpts:
        return "none"
    latest = max(ckpts, key=lambda p: p.stat().st_mtime)
    return latest.name


def tail_text(path: Path, lines: int = 8) -> str:
    if not path.exists():
        return "[log missing]"
    result = subprocess.run(
        ["tail", "-n", str(lines), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    text = result.stdout.strip()
    return text if text else "[log empty]"


def main():
    parser = argparse.ArgumentParser(description="Periodically log the status of a training PID.")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--interval-sec", type=int, default=7200)
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--train-log", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--label", type=str, default="training")
    args = parser.parse_args()

    log_file = Path(args.log_file)
    train_log = Path(args.train_log)
    ckpt_dir = Path(args.ckpt_dir)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{dt.datetime.utcnow().isoformat()}Z] monitor start\tlabel={args.label}\tpid={args.pid}\n")
        f.flush()

        while True:
            alive = pid_alive(args.pid)
            ts = dt.datetime.utcnow().isoformat() + "Z"
            ckpt = newest_checkpoint(ckpt_dir)
            f.write(f"[{ts}] alive={alive}\tpid={args.pid}\tlatest_ckpt={ckpt}\n")
            f.write(tail_text(train_log, lines=8) + "\n")
            f.write("-" * 80 + "\n")
            f.flush()

            if not alive:
                break
            time.sleep(max(1, args.interval_sec))

        f.write(f"[{dt.datetime.utcnow().isoformat()}Z] monitor stop\tlabel={args.label}\tpid={args.pid}\n")
        f.flush()


if __name__ == "__main__":
    main()
