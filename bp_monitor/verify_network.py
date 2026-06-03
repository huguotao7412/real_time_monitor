"""End-to-end verification: compare Python BP network output vs MATLAB reference.

Usage:
    python -m bp_monitor.verify_network

The script loads bp_matlab/bp_weights.mat and runs a self-consistency check.
For full MATLAB comparison, first export a reference pair from MATLAB:

    % In MATLAB, from bp_matlab/ directory:
    rng(42);
    test_input = randn(1, 256) * 0.01;
    test_input_norm = (test_input - x_min) / ((x_max - x_min) + 1e-8);
    dlX = dlarray(reshape(test_input_norm, 1, 1, 256), 'CBT');
    dlY = predict(bp_net, dlX);
    test_output = extractdata(gather(dlY));
    save('verify_ref.mat', 'test_input', 'test_output', 'x_min', 'x_max', 'y_min', 'y_max');
"""

import sys
import os
import numpy as np
from scipy.io import loadmat

from bp_monitor.bp_network import BPInference


def self_consistency_check(bp: BPInference) -> dict:
    """Run inference on zero, ones, and random inputs; check output sanity."""
    results = {}

    zero_out = bp.predict(np.zeros(256, dtype=np.float32))
    results["zero_min"] = float(np.min(zero_out))
    results["zero_max"] = float(np.max(zero_out))

    ones_out = bp.predict(np.ones(256, dtype=np.float32))
    results["ones_min"] = float(np.min(ones_out))
    results["ones_max"] = float(np.max(ones_out))

    rng = np.random.RandomState(42)
    rand_in = rng.randn(256).astype(np.float32) * 0.01
    rand_out = bp.predict(rand_in)
    results["rand_min"] = float(np.min(rand_out))
    results["rand_max"] = float(np.max(rand_out))
    results["rand_range"] = float(np.max(rand_out) - np.min(rand_out))

    return results


def compare_with_matlab(bp: BPInference, ref_path: str) -> dict:
    """Compare Python inference output with MATLAB reference.

    Args:
        bp: Loaded BPInference instance.
        ref_path: Path to verify_ref.mat exported from MATLAB.

    Returns:
        Dict with MSE, max_error, and per-sample error array.
    """
    ref = loadmat(ref_path, simplify_cells=True)
    test_input = np.asarray(ref["test_input"], dtype=np.float32).ravel()
    test_output_ref = np.asarray(ref["test_output"], dtype=np.float32).ravel()

    test_output_py = bp.predict(test_input)

    error = test_output_py - test_output_ref
    mse = float(np.mean(error ** 2))
    max_err = float(np.max(np.abs(error)))

    return {
        "mse": mse,
        "max_error": max_err,
        "error": error,
        "pass": mse < 1e-3,
    }


def main() -> int:
    print("=== BP Network Verification ===\n")

    weights_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bp_matlab", "bp_weights.mat"
    )
    if not os.path.exists(weights_path):
        print(f"[FAIL] Weights file not found: {weights_path}")
        return 1

    try:
        bp = BPInference(weights_path)
        print(f"[OK] Network loaded from {weights_path}")
    except Exception as e:
        print(f"[FAIL] Network load failed: {e}")
        return 1

    # Self-consistency check
    print("\n--- Self-Consistency Check ---")
    results = self_consistency_check(bp)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    checks_pass = True
    if results["rand_range"] < 0.001:
        print("[WARN] Random input produced near-constant output — check weights!")
        checks_pass = False
    else:
        print("[OK] Output range is plausible")

    # Optional MATLAB comparison
    ref_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bp_matlab", "verify_ref.mat"
    )
    if os.path.exists(ref_path):
        print(f"\n--- MATLAB Comparison ({ref_path}) ---")
        comp = compare_with_matlab(bp, ref_path)
        print(f"  MSE:       {comp['mse']:.6e}")
        print(f"  Max Error: {comp['max_error']:.6e}")
        if comp["pass"]:
            print("[OK] Output matches MATLAB reference (MSE < 1e-3)")
        else:
            print("[FAIL] Output differs from MATLAB reference!")
            checks_pass = False
    else:
        print(f"\n[SKIP] No MATLAB reference file at {ref_path}")
        print("  To enable: export verify_ref.mat from MATLAB (see script header)")

    return 0 if checks_pass else 1


if __name__ == "__main__":
    sys.exit(main())
