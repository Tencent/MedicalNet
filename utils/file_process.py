#!/usr/bin/env python
# coding: utf-8
"""Load a root-confined, validated image manifest."""
from utils.path_security import safe_image_path, safe_manifest_path


def load_lines(file_path, *, root_dir):
    """Read UTF-8 rows containing one image or an image/label pair.

    The required root_dir prevents callers from silently bypassing containment.
    All paths are checked before a row is returned; consumers recheck at use.
    """
    manifest = safe_manifest_path(root_dir, file_path)
    with open(manifest, 'r', encoding='utf-8-sig') as stream:
        lines = []
        for number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) not in (1, 2):
                raise ValueError(f'Manifest row {number} must have one or two paths.')
            for field in fields:
                safe_image_path(root_dir, field)
            lines.append(' '.join(fields))
    if not lines:
        raise ValueError('The image list must not be empty.')
    return lines
