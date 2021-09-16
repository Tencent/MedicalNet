#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp

def load_lines(file_path):
    """Read file into a list of lines.

    Input
      file_path: file path

    Output
      lines: an array of lines
    """
    with open(file_path, 'r') as fio:
        lines = fio.read().splitlines()
    return lines
