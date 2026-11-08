# Run the GPU port on MJX-Warp only; drop MJX-JAX and the model variant

The GPU port was planned as one MJX implementation switching engines by `impl="jax"` /
`impl="warp"`, with MJX-JAX running a **model variant** (robot collision meshes replaced by
convex hulls or primitives) because it cannot compile the base model's mesh pairs (8 GB host OOM
with two robots). The benchmark (`.scratch/gpu-port/research/05-mjx-benchmark.md`) showed that
even with every robot collision mesh replaced by one convex prism, MJX-JAX runs 40–50× *slower*
than the CPU backend with 12 threads, and its cylinder SDF loses a third of the side contacts
between ball and dribbler bar, which possession and the kick depend on. No reasonable
simplification closes a factor of 40. MJX-Warp runs the base model exactly (bar contact within
1e-5 mm of the CPU), and its physics ceiling is 25× / 15× the current CPU training pipeline end to
end (2 / 7 robots). So the port targets MJX-Warp alone, still through the `mujoco-mjx` JAX API.

## Consequences

- No model variant: no transform in `scene.build`, no kick table recalibrated for a variant, no
  variant × base parity check. Parity is one check: MJX-Warp × CPU on the base model.
- The MJX backend needs a CUDA GPU; tests without a GPU fall back to the CPU backend.
- No autodiff through the physics (MJX-Warp has none); differentiable simulation was already out
  of scope.
- No speedup threshold gates the port: all six tasks go to MJX-Warp. The spec ports
  ContestedPossession and GoToBall first and records end-to-end decisions/s against the CPU
  pipeline baseline, as a report, not a gate.
