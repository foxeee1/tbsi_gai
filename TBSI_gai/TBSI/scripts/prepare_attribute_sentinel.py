import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "data" / "lasher"
ATTR_ZIP = DATA_ROOT / "Attributes&Order.zip"
TEST_LIST = DATA_ROOT / "testingsetList.txt"
OUT_DIR = ROOT / "output" / "diagnostics" / "attribute_sentinel"
SEQ_LIST_OUT = ROOT / "experiments" / "tbsi_track" / "attribute_sentinel_sequences.txt"

TARGET_ATTRS = ["NO", "HO", "LI", "BC", "LR", "CM", "FM", "TC", "PO", "SV"]
SUPPORT_ATTRS = ["HI", "ARV", "AIV", "OV", "FL"]
MAX_SEQUENCES = 72
MIN_PER_TARGET = 8
SKIP = {"advancedredcup", "cameraman_1202", "mirroratleft"}


def parse_attrs():
    test_sequences = [s.strip() for s in TEST_LIST.read_text().splitlines() if s.strip()]
    test_sequences = [s for s in test_sequences if s not in SKIP]
    test_set = set(test_sequences)

    with zipfile.ZipFile(ATTR_ZIP) as zf:
        order = zf.read("Attributes_order.txt").decode().strip().replace(" ", "")
        attr_names = [x for x in order.split(",") if x]
        seq_attrs = {}
        for seq in test_sequences:
            name = f"AttriSeqsTxt/{seq}.txt"
            if name not in zf.namelist():
                continue
            values = [int(x) for x in zf.read(name).decode().strip().split(",")]
            attrs = [attr for attr, value in zip(attr_names, values) if value == 1]
            seq_attrs[seq] = attrs
    return attr_names, {s: seq_attrs[s] for s in test_sequences if s in seq_attrs and s in test_set}


def select_sentinel(seq_attrs):
    target = set(TARGET_ATTRS)
    selected = []
    selected_set = set()
    counts = Counter()

    def attr_score(seq):
        attrs = set(seq_attrs[seq])
        need = sum(1 for a in attrs & target if counts[a] < MIN_PER_TARGET)
        support = sum(1 for a in attrs & set(SUPPORT_ATTRS))
        rare_bonus = sum(1.0 / (1 + counts[a]) for a in attrs & target)
        return (need, rare_bonus, support, len(attrs), -len(selected))

    while len(selected) < MAX_SEQUENCES:
        if all(counts[a] >= MIN_PER_TARGET for a in TARGET_ATTRS):
            break
        candidates = [s for s in seq_attrs if s not in selected_set]
        if not candidates:
            break
        best = max(candidates, key=attr_score)
        if attr_score(best)[0] == 0:
            break
        selected.append(best)
        selected_set.add(best)
        counts.update(a for a in seq_attrs[best] if a in TARGET_ATTRS)

    if len(selected) < MAX_SEQUENCES:
        candidates = [s for s in seq_attrs if s not in selected_set]
        candidates.sort(key=lambda s: (
            sum(1 for a in seq_attrs[s] if a in TARGET_ATTRS),
            sum(1 for a in seq_attrs[s] if a in SUPPORT_ATTRS),
            len(seq_attrs[s]),
        ), reverse=True)
        for seq in candidates:
            selected.append(seq)
            selected_set.add(seq)
            counts.update(a for a in seq_attrs[seq] if a in TARGET_ATTRS)
            if len(selected) >= MAX_SEQUENCES:
                break

    return selected, counts


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    attr_names, seq_attrs = parse_attrs()
    selected, target_counts = select_sentinel(seq_attrs)

    attr_to_seqs = defaultdict(list)
    for seq in selected:
        for attr in seq_attrs[seq]:
            attr_to_seqs[attr].append(seq)

    SEQ_LIST_OUT.write_text("\n".join(selected) + "\n")
    manifest = {
        "sequence_count": len(selected),
        "target_attrs": TARGET_ATTRS,
        "support_attrs": SUPPORT_ATTRS,
        "max_sequences": MAX_SEQUENCES,
        "min_per_target": MIN_PER_TARGET,
        "attribute_order": attr_names,
        "target_counts": {attr: target_counts[attr] for attr in TARGET_ATTRS},
        "all_counts_in_sentinel": {attr: len(attr_to_seqs[attr]) for attr in attr_names},
        "sequences": selected,
        "sequence_attributes": {seq: seq_attrs[seq] for seq in selected},
        "list_path": str(SEQ_LIST_OUT.relative_to(ROOT)),
    }
    out_path = OUT_DIR / "attribute_sentinel_manifest.json"
    json.dump(manifest, out_path.open("w"), indent=2, ensure_ascii=False)
    print(json.dumps({
        "wrote": str(out_path),
        "list": str(SEQ_LIST_OUT),
        "sequence_count": len(selected),
        "target_counts": manifest["target_counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
