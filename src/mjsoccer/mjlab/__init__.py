"""The mjlab backend: the tasks a third time, in torch, over mjlab (ADR 0003).

`base` and `tasks` hold the task logic as functions over tensors: the physical state
(qpos, qvel, xpos, xmat, sensordata, batched over worlds, as mjlab's `sim.data` hands them
out) and a per-world state dict, the torch twin of `State.info["world"]` in `mjsoccer.mjx`.
They never touch a simulator, so tests/test_mjlab_task_logic.py holds them to the CPU
backend state by state in torch on the CPU. Every constant and task config is read off the
CPU side. Needs torch (`uv sync --group mjlab`); `import mjsoccer` never imports this package.
"""
