"""Run all five scripts, then all five notebooks, or either form separately."""
from pathlib import Path
import os
import sys
import subprocess
import argparse
import hashlib
import json
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
from sync_notebooks import script_cells

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=['all', 'scripts', 'notebooks'], default='all')
parser.add_argument('--part', type=int, choices=range(1, 6), help='Run only this stage; used by run_stages.py.')
args = parser.parse_args()
if args.part in [None, 1]:
    part1_notebook = nbformat.read(ROOT/'notebooks/01_data_preparation.ipynb', as_version=4)
    part1_cells = script_cells(ROOT/'scripts/part_01_data_preparation.py')
    assert [(c.cell_type, c.source) for c in part1_notebook.cells] == [(c.cell_type, c.source) for c in part1_cells], \
        'Part 1 notebook differs from its script. Run src/sync_notebooks.py before execution.'
runtime = ROOT / 'outputs' / 'jupyter_runtime'
runtime.mkdir(parents=True, exist_ok=True)
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime)
os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
os.environ['MPLCONFIGDIR'] = str(runtime / 'matplotlib')

if args.mode in ['all', 'scripts']:
    scripts = sorted((ROOT / 'scripts').glob('part_*.py'))
    if args.part:
        scripts = [scripts[args.part-1]]
    for script in scripts:
        print('Running script:', script.name, flush=True)
        environment = os.environ.copy()
        environment['MPLBACKEND'] = 'Agg'
        subprocess.run([sys.executable, str(script)], cwd=ROOT, env=environment, check=True)
    if args.mode == 'scripts':
        sys.exit(0)
    csv_paths = list((ROOT/'data/processed').glob('*.csv')) + list((ROOT/'outputs/tables').glob('*.csv'))
    script_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in csv_paths}

class ProjectKernelSpecManager(KernelSpecManager):
    def get_kernel_spec(self, kernel_name):
        spec = super().get_kernel_spec(kernel_name)
        spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        spec.metadata['supported_encryption'] = ['curve']
        return spec

notebook_names = ['01_data_preparation.ipynb', '02_feature_engineering.ipynb',
                  '03_volatility_forecasting.ipynb', '04_var_backtesting.ipynb', '05_results_and_discussion.ipynb']
if args.part:
    notebook_names = [notebook_names[args.part-1]]
for name in notebook_names:
    path = ROOT / 'notebooks' / name
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=300, kernel_name='python3',
                            resources={'metadata': {'path': str(ROOT)}},
                            kernel_manager_class='jupyter_client.manager.KernelManager')
    client.create_kernel_manager()
    client.km.kernel_spec_manager = ProjectKernelSpecManager()
    client.km.transport_encryption = 'required'
    client.on_cell_executed = lambda **kwargs: nbformat.write(notebook, path)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    print('Executed successfully:', name, flush=True)

if args.mode == 'all':
    for name, digest in script_hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, f'Script/notebook CSV mismatch: {name}'
    parity_path = ROOT/'outputs/tables/execution_parity.json'
    if args.part:
        parity_path = ROOT/f'outputs/checkpoints/part_{args.part}_parity.json'
        parity_path.parent.mkdir(parents=True, exist_ok=True)
    parity_path.write_text(json.dumps({
        'all_five_scripts_executed': args.part is None, 'all_five_notebooks_executed': args.part is None,
        'csv_files_byte_identical': len(script_hashes), 'csv_sha256': script_hashes}, indent=2))
    print('All scripts and notebooks produced byte-identical CSVs.', flush=True)
