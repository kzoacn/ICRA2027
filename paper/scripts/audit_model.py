#!/usr/bin/env python3
"""Count the pinned detector's parameters on CPU, without running robot tasks."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import platform
import sys

PAPER = Path(__file__).resolve().parents[1]
IMPLEMENTATION = PAPER.parent / "src"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Local grounding-dino-tiny directory from resources.lock.json.")
    parser.add_argument("--output", type=Path, default=PAPER / "data/model_audit.json")
    args = parser.parse_args()
    checkpoint = json.loads((PAPER.parent / "configs/resources.lock.json").read_text())["model"]
    for expected in checkpoint["files"]:
        path = args.model_dir / expected["path"]
        if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Checkpoint file does not match resource lock: {path.name}")

    sys.path.insert(0, str(IMPLEMENTATION))
    import torch
    import transformers
    from anchor.perception.grounding_dino import GroundingDINOBoxDetector

    detector = GroundingDINOBoxDetector(args.model_dir, device="cpu", local_files_only=True)
    # Module.parameters() removes shared Parameter objects by default.
    parameters = list(detector.model.parameters())
    dtypes = collections.Counter()
    for parameter in parameters:
        dtypes[str(parameter.dtype)] += parameter.numel()
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    if trainable or detector.model.training:
        raise ValueError("The deployed detector must remain frozen in evaluation mode.")
    implementation_file = IMPLEMENTATION / "anchor/perception/grounding_dino.py"
    weight_file = next(item for item in checkpoint["files"]
                       if item["path"] == "model.safetensors")
    result = {
        "schema": 1,
        "checkpoint": checkpoint,
        "implementation": {
            "class": "GroundingDINOBoxDetector",
            "file": str(implementation_file.relative_to(PAPER.parent)),
            "sha256": sha256(implementation_file),
        },
        "counting_method": "Sum numel() over unique model.parameters() after loading the deployed class.",
        "scope": "Entire frozen detector, including vision and text encoders; no robot rollout or latency benchmark.",
        "parameters": sum(parameter.numel() for parameter in parameters),
        "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in parameters),
        "runtime_trainable_parameters": trainable,
        "parameter_elements_by_dtype": dict(dtypes),
        "weight_file": weight_file,
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"parameters": result["parameters"], "trainable": trainable,
                      "weight_bytes": weight_file["size"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
