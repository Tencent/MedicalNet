"""Filesystem boundaries for manifests and NIfTI inputs.

The root directory is trusted operator configuration, not an untrusted request
parameter. Its directory tree must not be writable by concurrent attackers.
"""
import ntpath
import os
from pathlib import Path


def _contained_file(root_dir, candidate, suffixes):
    root = Path(root_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Data root must be a directory.')
    resolved = Path(candidate).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError('Input file is outside the data root.') from exc
    if not resolved.is_file():
        raise ValueError('Input must be a regular file.')
    if not resolved.name.lower().endswith(suffixes):
        raise ValueError('Unsupported input file extension.')
    return str(resolved)


def safe_image_path(root_dir, relative_path):
    """Accept only relative .nii/.nii.gz files resolving inside root_dir."""
    raw = os.fspath(relative_path)
    if not isinstance(raw, str) or not raw or '\x00' in raw:
        raise ValueError('An image path must be a nonempty text path.')
    normalized = raw.replace('\\', '/')
    if (ntpath.splitdrive(raw)[0] or normalized.startswith('/') or
            ':' in normalized or '..' in normalized.split('/')):
        raise ValueError('Absolute, drive, stream and parent paths are forbidden.')
    if not normalized.lower().endswith(('.nii', '.nii.gz')):
        raise ValueError('Only .nii and .nii.gz images are supported.')
    return _contained_file(root_dir, Path(root_dir) / normalized, ('.nii', '.nii.gz'))


def safe_manifest_path(root_dir, file_path):
    """CLI manifest paths remain cwd-relative (or absolute), but root-confined."""
    raw = os.fspath(file_path)
    if not isinstance(raw, str) or not raw or '\x00' in raw:
        raise ValueError('A manifest path must be a nonempty text path.')
    drive, tail = ntpath.splitdrive(raw)
    if ':' in tail or (drive and not ntpath.isabs(raw)):
        raise ValueError('Stream and drive-relative manifest paths are forbidden.')
    if not raw.lower().endswith('.txt'):
        raise ValueError('The image list must be a .txt file.')
    return _contained_file(root_dir, raw, ('.txt',))
