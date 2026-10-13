"""Security regression tests; only generated fixtures inside TemporaryDirectory.

Run: python -m unittest discover -s tests -v
No network requests, subprocess payloads, or real secrets are used.
"""
import ast
from collections import OrderedDict
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from datasets.brains18 import BrainS18Dataset
from model import generate_model
from utils.checkpoint import load_checkpoint
from utils.file_process import load_lines
from utils.path_security import safe_image_path, safe_manifest_path

ROOT = Path(__file__).resolve().parents[1]
CALLED = []


def harmless_marker():
    CALLED.append('deserializer invoked a Python callable')
    return {}


class CallableFixture:
    def __reduce__(self):
        return harmless_marker, ()


class PathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='medicalnet-path-')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'data'
        self.root.mkdir()
        (self.root / 'images').mkdir()
        self.image = self.root / 'images' / 'valid.nii.gz'
        data = np.arange(14 * 28 * 28, dtype=np.float32).reshape(14, 28, 28)
        nib.save(nib.Nifti1Image(data, np.eye(4)), self.image)
        self.label = self.root / 'images' / 'label.nii.gz'
        nib.save(nib.Nifti1Image((data > 50).astype(np.int16), np.eye(4)), self.label)
        self.outside = self.base / 'outside.nii.gz'
        nib.save(nib.Nifti1Image(data, np.eye(4)), self.outside)
        self.manifest = self.root / 'list.txt'
        self.manifest.write_text('images/valid.nii.gz images/label.nii.gz\n', encoding='utf-8')
        self.sets = SimpleNamespace(input_D=14, input_H=28, input_W=28, phase='test')

    def test_valid_relative_image_and_label(self):
        self.assertEqual(safe_image_path(self.root, 'images/valid.nii.gz'), str(self.image.resolve()))
        self.assertEqual(safe_image_path(self.root, 'images\\label.nii.gz'), str(self.label.resolve()))

    def test_parent_paths_both_separators_rejected(self):
        for name in ('../outside.nii.gz', '..\\outside.nii.gz', 'images/../../outside.nii.gz'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_image_path(self.root, name)

    def test_absolute_drive_unc_and_stream_paths_rejected(self):
        for name in (str(self.image), str(self.outside), 'C:relative.nii.gz', '/tmp/a.nii',
                     'C:\\outside.nii', '\\\\server\\share\\a.nii', 'images/valid.nii.gz:stream.nii'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_image_path(self.root, name)

    def test_suffix_missing_directory_and_nul_rejected(self):
        (self.root / 'not_image.txt').write_text('fixture', encoding='utf-8')
        (self.root / 'directory.nii').mkdir()
        for name in ('not_image.txt', 'missing.nii.gz', 'directory.nii', '\x00.nii', ''):
            with self.subTest(name=name), self.assertRaises((ValueError, OSError)):
                safe_image_path(self.root, name)

    def test_manifest_inside_root_allowed(self):
        self.assertEqual(safe_manifest_path(self.root, self.manifest), str(self.manifest.resolve()))
        relative = os.path.relpath(self.manifest, start=self.base)
        original_cwd = os.getcwd()
        try:
            os.chdir(self.base)
            self.assertEqual(len(load_lines(relative, root_dir=self.root)), 1)
        finally:
            os.chdir(original_cwd)

    def test_manifest_outside_and_prefix_sibling_rejected(self):
        sibling = self.base / 'data-evil'
        sibling.mkdir()
        for path in (self.base / 'outside.txt', sibling / 'list.txt'):
            path.write_text('images/valid.nii.gz', encoding='utf-8')
            with self.subTest(path=path), self.assertRaises(ValueError):
                load_lines(path, root_dir=self.root)

    def test_manifest_traversal_in_either_column_rejected_before_nib_load(self):
        for row in ('../outside.nii.gz images/label.nii.gz',
                    'images/valid.nii.gz ../outside.nii.gz'):
            self.manifest.write_text(row, encoding='utf-8')
            with patch('nibabel.load') as mocked, self.assertRaises(ValueError):
                BrainS18Dataset(self.root, self.manifest, self.sets)
            mocked.assert_not_called()

    def test_blank_lines_and_tabs_are_normalized(self):
        self.manifest.write_text('\ufeff\nimages/valid.nii.gz\timages/label.nii.gz\n\n', encoding='utf-8')
        self.assertEqual(load_lines(self.manifest, root_dir=self.root),
                         ['images/valid.nii.gz images/label.nii.gz'])

    def test_empty_and_malformed_rows_rejected(self):
        for row in ('\n', 'images/valid.nii.gz images/label.nii.gz extra.nii'):
            self.manifest.write_text(row, encoding='utf-8')
            with self.assertRaises(ValueError):
                load_lines(self.manifest, root_dir=self.root)

    def test_test_dataset_real_nifti(self):
        dataset = BrainS18Dataset(self.root, self.manifest, self.sets)
        volume = dataset[0]
        self.assertEqual(volume.shape, (1, 14, 28, 28))
        self.assertTrue(np.isfinite(volume).all())

    def test_train_dataset_real_nifti(self):
        self.sets.phase = 'train'
        dataset = BrainS18Dataset(self.root, self.manifest, self.sets)
        volume, label = dataset[0]
        self.assertEqual(volume.shape, label.shape)
        self.assertEqual(volume.shape, (1, 14, 28, 28))

    def test_train_requires_label(self):
        self.manifest.write_text('images/valid.nii.gz', encoding='utf-8')
        self.sets.phase = 'train'
        dataset = BrainS18Dataset(self.root, self.manifest, self.sets)
        with self.assertRaises(ValueError):
            dataset[0]

    def test_junction_or_symlink_escape_rejected(self):
        outside_dir = self.base / 'outside_dir'
        outside_dir.mkdir()
        (outside_dir / 'image.nii.gz').write_bytes(self.outside.read_bytes())
        (outside_dir / 'list.txt').write_text('images/valid.nii.gz', encoding='utf-8')
        link = self.root / 'link'
        if os.name == 'nt':
            result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(outside_dir)],
                                    capture_output=True, text=True)
            if result.returncode:
                self.skipTest('Unable to create local test junction: ' + result.stderr)
        else:
            link.symlink_to(outside_dir, target_is_directory=True)
        try:
            with self.assertRaises(ValueError):
                safe_image_path(self.root, 'link/image.nii.gz')
            with self.assertRaises(ValueError):
                safe_manifest_path(self.root, link / 'list.txt')
        finally:
            if os.name == 'nt':
                os.rmdir(link)
            else:
                link.unlink()


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='medicalnet-checkpoint-')
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / 'checkpoint.pth'
        CALLED.clear()
        self.env = patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop('TORCH_FORCE_WEIGHTS_ONLY_LOAD', None)
        os.environ.pop('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', None)

    def save(self, checkpoint):
        torch.save(checkpoint, self.file)

    def test_valid_tensor_checkpoint_and_cpu_mapping(self):
        self.save({'state_dict': OrderedDict(weight=torch.ones(2))})
        result = load_checkpoint(self.file)
        self.assertEqual(result['state_dict']['weight'].device.type, 'cpu')
        self.assertTrue(torch.equal(result['state_dict']['weight'], torch.ones(2)))

    def test_optimizer_round_trip(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        self.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': 1})
        result = load_checkpoint(self.file, require_optimizer=True)
        optimizer.load_state_dict(result['optimizer'])
        self.assertEqual(result['epoch'], 1)

    def test_unsafe_callable_rejected_without_execution(self):
        self.save({'state_dict': CallableFixture()})
        with self.assertRaises(pickle.UnpicklingError):
            load_checkpoint(self.file)
        self.assertEqual(CALLED, [])

    def test_environment_opt_out_cannot_disable_explicit_weights_only(self):
        self.save({'state_dict': CallableFixture()})
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
        with self.assertRaises(pickle.UnpicklingError):
            load_checkpoint(self.file)
        self.assertEqual(CALLED, [])

    def test_no_unsafe_retry(self):
        self.file.write_bytes(b'fixture')
        with patch('torch.load', side_effect=TypeError('unsupported')) as loader:
            with self.assertRaises(TypeError):
                load_checkpoint(self.file)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(loader.call_args.kwargs, {'map_location': 'cpu', 'weights_only': True})

    def test_old_runtime_rejected_before_load(self):
        with patch.object(torch, '__version__', '2.5.1'), patch('torch.load') as loader:
            with self.assertRaises(RuntimeError):
                load_checkpoint(self.file)
            loader.assert_not_called()

    def test_custom_safe_globals_rejected(self):
        self.save({'state_dict': CallableFixture()})
        with torch.serialization.safe_globals([harmless_marker]):
            before = set(torch.serialization.get_safe_globals())
            with self.assertRaises(pickle.UnpicklingError):
                load_checkpoint(self.file)
            self.assertEqual(set(torch.serialization.get_safe_globals()), before)
        self.assertEqual(CALLED, [])

    def test_schema_and_optimizer_validation(self):
        for content in ([], {}, {'state_dict': []}, {'state_dict': {'weight': 'not a tensor'}},
                        {'state_dict': {1: torch.ones(1)}}):
            self.save(content)
            with self.subTest(content=repr(content)), self.assertRaises(ValueError):
                load_checkpoint(self.file)
        self.save({'state_dict': {}})
        with self.assertRaises(ValueError):
            load_checkpoint(self.file, require_optimizer=True)

    def test_corrupt_and_missing_files_fail(self):
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(self.file)
        self.file.write_bytes(b'not a checkpoint')
        with self.assertRaises(Exception):
            load_checkpoint(self.file)

    def test_pretrained_model_entry_and_forward(self):
        sets = SimpleNamespace(model='resnet', model_depth=10, input_W=28, input_H=28,
                               input_D=14, resnet_shortcut='B', no_cuda=True,
                               n_seg_classes=2, phase='test', pretrain_path='',
                               new_layer_names=['conv_seg'])
        model, _ = generate_model(sets)
        self.save({'state_dict': model.state_dict()})
        sets.phase = 'train'
        sets.pretrain_path = str(self.file)
        restored, groups = generate_model(sets)
        self.assertIn('base_parameters', groups)
        restored.eval()
        with torch.no_grad():
            result = restored(torch.ones(1, 1, 14, 28, 28))
        self.assertEqual(tuple(result.shape[:2]), (1, 2))
        self.save({'state_dict': CallableFixture()})
        with self.assertRaises(pickle.UnpicklingError):
            generate_model(sets)
        self.assertEqual(CALLED, [])


class WiringTests(unittest.TestCase):
    def test_only_central_loader_calls_torch_load(self):
        calls = []
        for path in ROOT.rglob('*.py'):
            if 'tests' in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and ast.unparse(node.func) == 'torch.load':
                    calls.append(path.relative_to(ROOT).as_posix())
                    self.assertIn('weights_only=True', ast.unparse(node))
        self.assertEqual(calls, ['utils/checkpoint.py'])
        for name in ('train.py', 'test.py', 'model.py'):
            self.assertIn('load_checkpoint(', (ROOT / name).read_text(encoding='utf-8'))

    def test_all_nifti_calls_are_root_confined(self):
        for name in ('datasets/brains18.py', 'test.py'):
            text = (ROOT / name).read_text(encoding='utf-8')
            self.assertNotIn('os.path.join(', text)
            self.assertIn('safe_image_path(', text)


if __name__ == '__main__':
    unittest.main(verbosity=2)