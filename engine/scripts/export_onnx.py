#!/usr/bin/env python3
"""Export RazzleNet to ONNX format for browser inference.

Usage:
    python scripts/export_onnx.py --checkpoint path/to/model.pt --output model.onnx
    python scripts/export_onnx.py --random-weights --preset small --output model.onnx
    python scripts/export_onnx.py --random-weights --preset small --output model.onnx --quantize
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add engine to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from razzle.ai.network import RazzleNet, NetworkConfig, PRESETS, NUM_ACTIONS
from razzle.core.bitboard import ROWS, COLS


def export_onnx(
    model: RazzleNet,
    output_path: str,
    quantize: bool = False,
    verify: bool = True,
) -> None:
    """Export a RazzleNet model to ONNX format.

    Args:
        model: The PyTorch model to export.
        output_path: Path for the .onnx output file.
        quantize: Apply dynamic quantization for smaller file size.
        verify: Verify exported model matches PyTorch output.
    """
    model.eval()

    # Dummy input: (batch=1, channels=7, rows=8, cols=7)
    dummy_input = torch.randn(1, 7, ROWS, COLS)

    # Get PyTorch reference output before export
    with torch.no_grad():
        ref_policy, ref_value, ref_difficulty = model(dummy_input)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["board_input"],
        output_names=["policy", "value", "difficulty"],
        dynamic_axes={
            "board_input": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
            "difficulty": {0: "batch"},
        },
    )

    file_size = Path(output_path).stat().st_size
    print(f"Exported ONNX model: {file_size / 1024:.1f} KB")

    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_path = output_path.replace(".onnx", "_quantized.onnx")
            quantize_dynamic(
                output_path,
                quantized_path,
                weight_type=QuantType.QUInt8,
            )
            q_size = Path(quantized_path).stat().st_size
            print(f"Quantized model: {q_size / 1024:.1f} KB ({q_size / file_size:.1%} of original)")
        except ImportError:
            print("Warning: onnxruntime not installed, skipping quantization")

    if verify:
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(output_path)
            onnx_out = sess.run(None, {"board_input": dummy_input.numpy()})
            onnx_policy, onnx_value, onnx_difficulty = onnx_out

            policy_err = np.max(np.abs(ref_policy.numpy() - onnx_policy))
            value_err = np.max(np.abs(ref_value.numpy() - onnx_value))
            diff_err = np.max(np.abs(ref_difficulty.numpy() - onnx_difficulty))

            print(f"Verification - max errors: policy={policy_err:.2e}, value={value_err:.2e}, difficulty={diff_err:.2e}")

            if policy_err > 1e-5 or value_err > 1e-5 or diff_err > 1e-5:
                print("WARNING: ONNX output differs from PyTorch by more than 1e-5!")
                return

            print("Verification PASSED")
        except ImportError:
            print("Warning: onnxruntime not installed, skipping verification")


def main():
    parser = argparse.ArgumentParser(description="Export RazzleNet to ONNX")
    parser.add_argument("--checkpoint", type=str, help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX file path")
    parser.add_argument("--random-weights", action="store_true", help="Export model with random weights (for testing)")
    parser.add_argument("--preset", type=str, default="small", choices=list(PRESETS.keys()), help="Model size preset")
    parser.add_argument("--quantize", action="store_true", help="Apply dynamic quantization")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")

    args = parser.parse_args()

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model = RazzleNet.load(args.checkpoint)
    elif args.random_weights:
        print(f"Creating model with random weights (preset: {args.preset})")
        config = PRESETS[args.preset]
        model = RazzleNet(config)
    else:
        parser.error("Must specify --checkpoint or --random-weights")

    print(f"Model: {model.config}")
    print(f"Parameters: {model.num_parameters():,}")

    export_onnx(
        model,
        args.output,
        quantize=args.quantize,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
