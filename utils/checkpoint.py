"""Fail-closed loading for tensor state-dict checkpoints.

Use trusted model sources. weights_only is not a memory/CPU resource sandbox.
Run each job in a dedicated process: unrelated threads must not concurrently
register globals or call torch.load while this loader isolates the registry.
"""
from collections.abc import Mapping
from pathlib import Path
import re
import threading
import torch

_LOAD_LOCK = threading.RLock()


def load_checkpoint(file_path, *, require_optimizer=False):
    """Load tensor weights on CPU, without custom globals or unsafe fallback."""
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)', torch.__version__)
    if match is None or tuple(map(int, match.groups())) < (2, 6, 0):
        raise RuntimeError('Safe checkpoint loading requires PyTorch >= 2.6.0.')
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint file not found: {path}')
    # PyTorch can register extra globals itself. Isolate, then restore them;
    # do not mistake a nonempty registry for a malicious runtime.
    with _LOAD_LOCK:
        previous = torch.serialization.get_safe_globals()
        torch.serialization.clear_safe_globals()
        try:
            # Never retry with weights_only=False when an older file fails.
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        finally:
            torch.serialization.add_safe_globals(previous)
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Checkpoint must be a mapping with a state_dict.')
    state = checkpoint.get('state_dict')
    if not isinstance(state, Mapping):
        raise ValueError('Checkpoint state_dict must be a tensor mapping.')
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
           for key, value in state.items()):
        raise ValueError('state_dict must contain only string keys and tensors.')
    if require_optimizer:
        optimizer = checkpoint.get('optimizer')
        if (not isinstance(optimizer, Mapping) or
                not isinstance(optimizer.get('state'), Mapping) or
                not isinstance(optimizer.get('param_groups'), list)):
            raise ValueError('Resume checkpoint must include optimizer state.')
    return checkpoint
