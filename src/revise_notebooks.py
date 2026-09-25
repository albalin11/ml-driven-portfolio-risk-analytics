"""Compatibility entry point: scripts are now the source for notebook code.

The previous migration script is retained in the local pre-project backup.
"""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('sync_notebooks.py')), run_name='__main__')
