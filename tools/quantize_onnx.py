#!/usr/bin/env python3
"""Create a separately stored dynamic-int8 ONNX model for CPU experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


def quantize_model(source: Path, destination: Path, per_channel: bool = False) -> None:
    """Quantize an ONNX model without modifying the source file."""
    if not source.is_file():
        raise FileNotFoundError(f"ONNX model not found: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("The quantized model must use a different output path.")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    destination.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(source),
        model_output=str(destination),
        per_channel=per_channel,
        weight_type=QuantType.QInt8,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to the original ONNX model")
    parser.add_argument("output", type=Path, help="Path for the new int8 ONNX model")
    parser.add_argument("--per-channel", action="store_true", help="Quantize weights per channel")
    args = parser.parse_args()

    quantize_model(args.model, args.output, args.per_channel)
    source_size = args.model.stat().st_size / (1024 * 1024)
    output_size = args.output.stat().st_size / (1024 * 1024)
    print(f"source_mb={source_size:.2f} quantized_mb={output_size:.2f} reduction={1 - output_size / source_size:.1%}")


if __name__ == "__main__":
    main()
