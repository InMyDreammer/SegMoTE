

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint_utils import upgrade_legacy_state_dict


def convert_checkpoint(source: Path, destination: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = upgrade_legacy_state_dict(state_dict)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state_dict}, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Path to a legacy training checkpoint.")
    parser.add_argument("destination", type=Path, help="Path for the released inference checkpoint.")
    args = parser.parse_args()
    convert_checkpoint(args.source, args.destination)
    print(f"Saved released checkpoint to {args.destination}")


if __name__ == "__main__":
    main()
