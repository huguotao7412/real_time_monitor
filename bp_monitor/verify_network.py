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
import torch
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
    if results["rand_range"] < 5.0:
        print(f"[FAIL] Random input produced near-constant output "
              f"(range={results['rand_range']:.4f} mmHg, expected >5 mmHg). "
              f"Check weight loading!")
        checks_pass = False
    else:
        print(f"[OK] Output range is plausible ({results['rand_range']:.1f} mmHg)")

    # Compare zero vs random — outputs should differ
    zero_mean = float(np.mean(bp.predict(np.zeros(256, dtype=np.float32))))
    rand_mean = float(np.mean(bp.predict(np.random.RandomState(99).randn(256).astype(np.float32) * 0.01)))
    if abs(zero_mean - rand_mean) < 0.01:
        print(f"[FAIL] Zero and random inputs produce identical output "
              f"(zero_mean={zero_mean:.4f}, rand_mean={rand_mean:.4f}). "
              f"Network is ignoring input!")
        checks_pass = False
    else:
        print(f"[OK] Network responds to different inputs "
              f"(zero_mean={zero_mean:.2f}, rand_mean={rand_mean:.2f})")

    # BatchNorm diagnostic
    print("\n--- BatchNorm Diagnostic ---")
    bn_issues = []
    for name, mod in bp.model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            rv = mod.running_var.detach().numpy()
            rm = mod.running_mean.detach().numpy()
            rv_min, rv_max = float(np.min(rv)), float(np.max(rv))
            if rv_min < 1e-6:
                bn_issues.append(f"{name}: running_var min={rv_min:.2e} (near zero!)")
            if rv_max > 1e6:
                bn_issues.append(f"{name}: running_var max={rv_max:.2e} (extreme!)")
    if bn_issues:
        print("[FAIL] BatchNorm issues found:")
        for issue in bn_issues:
            print(f"  {issue}")
        checks_pass = False
    else:
        print("[OK] BatchNorm running stats look normal")

    # Layer activation trace — find where signal dies
    print("\n--- Activation Trace (zero vs random) ---")
    acts_zero = {}
    acts_rand = {}

    def hook_fn(name, d):
        def fn(module, input, output):
            if isinstance(output, torch.Tensor):
                d[name] = output.detach().cpu().numpy().copy()
            elif isinstance(output, (tuple, list)):
                d[name] = output[0].detach().cpu().numpy().copy()
        return fn

    key_names = ["input_proj", "enc1", "enc2", "enc3", "enc4",
                 "bridge_base", "bridge_aspp",
                 "dec4", "dec3", "dec2", "dec1",
                 "final_conv", "sigmoid",
                 "rams1", "rams2", "rams3", "rams4",
                 "rams_bridge", "rams_dec4", "rams_dec3", "rams_dec2", "rams_dec1"]

    hooks = []
    for name, mod in bp.model.named_modules():
        for kn in key_names:
            if name.endswith(kn) and not any(c.isdigit() for c in name.replace(kn, "").replace(".", "")):
                hooks.append(mod.register_forward_hook(hook_fn(name, acts_zero)))
                break

    x_norm = (np.zeros(256, dtype=np.float32) - bp.x_min) / bp.x_rng
    x_t = torch.from_numpy(x_norm).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        _ = bp.model(x_t)
    for h in hooks:
        h.remove()

    hooks = []
    for name, mod in bp.model.named_modules():
        for kn in key_names:
            if name.endswith(kn) and not any(c.isdigit() for c in name.replace(kn, "").replace(".", "")):
                hooks.append(mod.register_forward_hook(hook_fn(name, acts_rand)))
                break

    rng_np = np.random.RandomState(42)
    rand_in = rng_np.randn(256).astype(np.float32) * 0.01
    x_norm_r = (rand_in - bp.x_min) / bp.x_rng
    x_t_r = torch.from_numpy(x_norm_r).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        _ = bp.model(x_t_r)
    for h in hooks:
        h.remove()

    for name in sorted(acts_zero.keys()):
        z = acts_zero[name]
        r = acts_rand[name]
        z_std = float(np.std(z))
        r_std = float(np.std(r))
        diff = float(np.max(np.abs(z - r)))
        status = "DEAD" if (diff < 1e-6) else "OK"
        print(f"  {name:30s} zero_std={z_std:8.4f} rand_std={r_std:8.4f} diff={diff:8.6f} [{status}]")

    # Weight statistics — check for dead layers
    print("\n--- Weight Statistics ---")
    weight_issues = []
    for name, mod in bp.model.named_modules():
        if isinstance(mod, torch.nn.Conv1d):
            w = mod.weight.detach().numpy()
            b = mod.bias.detach().numpy() if mod.bias is not None else None
            w_std = float(np.std(w))
            w_range = float(np.max(np.abs(w)))
            if w_range < 1e-8:
                weight_issues.append(f"{name}: weight all-zero!")
            b_info = f" bias_range=[{float(np.min(b)):.2e},{float(np.max(b)):.2e}]" if b is not None else ""
            if "final_conv" in name or "input_proj" in name:
                print(f"  {name}: w_range={w_range:.4f} w_std={w_std:.4f}{b_info}")
    if weight_issues:
        print("[FAIL] Dead weight layers:")
        for issue in weight_issues:
            print(f"  {issue}")
        checks_pass = False
    else:
        print("[OK] All conv weights have non-zero values")

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
