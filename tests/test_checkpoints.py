"""Checkpoint invalidation and public-commit boundary checks; no training required."""
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import sys
import unittest
import subprocess
import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import run_stages as stages

class CheckpointTests(unittest.TestCase):
    def test_checkpoint_requires_unchanged_files_inputs_and_ancestor(self):
        state = dict(status='complete', files={'output': 'a'}, inputs={'input': 'b'}, commit='saved')
        with patch.object(stages, 'stage_files', return_value=['output']), \
             patch.object(stages, 'file_hashes', return_value={'output': 'a'}) as hashes, \
             patch.object(stages, 'input_hashes', return_value={'input': 'b'}) as inputs, \
             patch.object(stages.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as ancestry:
            self.assertTrue(stages.checkpoint_valid(1, state))
            hashes.return_value = {'output': 'changed'}
            self.assertFalse(stages.checkpoint_valid(1, state))
            hashes.return_value = {'output': 'a'}
            inputs.return_value = {'input': 'changed'}
            self.assertFalse(stages.checkpoint_valid(1, state))
            inputs.return_value = {'input': 'b'}
            ancestry.return_value = SimpleNamespace(returncode=1)
            self.assertFalse(stages.checkpoint_valid(1, state))

    def test_failed_running_and_missing_artifacts_are_not_skipped(self):
        for status in ['running', 'failed']:
            self.assertFalse(stages.checkpoint_valid(1, {'status':status}))
        with patch.object(stages, 'file_hashes', side_effect=FileNotFoundError):
            self.assertFalse(stages.checkpoint_valid(1, {'status':'complete', 'files':{}}))

    def test_checkpoint_write_keeps_complete_json_and_commit(self):
        with TemporaryDirectory() as directory, patch.object(stages, 'STATE_DIR', Path(directory)):
            stages.save_state(1, {'status':'running'})
            stages.save_state(1, {'status':'complete', 'commit':'example'})
            self.assertEqual(json.loads((Path(directory)/'part_1.json').read_text()),
                             {'status':'complete', 'commit':'example'})
            self.assertFalse((Path(directory)/'part_1.tmp').exists())

    def test_public_stage_lists_never_include_private_artifacts(self):
        for part in range(1, 6):
            for name in stages.public_files(part):
                self.assertFalse(name.startswith(('data/', 'outputs/models/')))
                self.assertNotIn(name, ['outputs/tables/test_predictions.csv',
                    'outputs/tables/risk_forecasts.csv', 'outputs/tables/nonoverlapping_risk_forecasts.csv'])

    def test_existing_staged_work_stops_automatic_commit(self):
        with patch.object(stages, 'git', return_value='unrelated.py'), \
             patch.object(stages.subprocess, 'run') as command:
            with self.assertRaises(AssertionError):
                stages.commit_stage(1)
            command.assert_not_called()

    def test_failed_execution_preserves_backup_and_never_commits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'scripts').mkdir()
            (root/'notebooks').mkdir()
            (root/'scripts/stage.py').write_text('# %%\nraise RuntimeError("example failure")\n')
            nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('print("saved")')]),
                           root/'notebooks/stage.ipynb')
            original = (root/'notebooks/stage.ipynb').read_bytes()
            with patch.object(stages, 'ROOT', root), \
                 patch.object(stages, 'STATE_DIR', root/'outputs/checkpoints'), \
                 patch.object(stages, 'NAMES', [('stage.py', 'stage.ipynb')]), \
                 patch.object(stages, 'OUTPUTS', {1: []}), \
                 patch.object(stages.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'python')), \
                 patch.object(stages, 'commit_stage') as commit:
                with self.assertRaises(subprocess.CalledProcessError):
                    stages.run_part(1)
                state = json.loads((root/'outputs/checkpoints/part_1.json').read_text())
                self.assertEqual(state['status'], 'failed')
                self.assertEqual((root/state['backup']/'notebooks/stage.ipynb').read_bytes(), original)
                commit.assert_not_called()

if __name__ == '__main__':
    unittest.main()
