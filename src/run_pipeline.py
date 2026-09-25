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

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=['all', 'scripts', 'notebooks'], default='all')
args = parser.parse_args()
runtime = ROOT / 'outputs' / 'jupyter_runtime'
runtime.mkdir(parents=True, exist_ok=True)
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime)
os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
os.environ['MPLCONFIGDIR'] = str(runtime / 'matplotlib')

if args.mode in ['all', 'scripts']:
    for script in sorted((ROOT / 'scripts').glob('part_*.py')):
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

for name in ['01_data_preparation.ipynb', '02_feature_engineering.ipynb',
             '03_volatility_forecasting.ipynb', '04_var_backtesting.ipynb', '05_results_and_discussion.ipynb']:
    path = ROOT / 'notebooks' / name
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=300, kernel_name='python3',
                            resources={'metadata': {'path': str(ROOT)}},
                            kernel_manager_class='jupyter_client.manager.KernelManager')
    client.create_kernel_manager()
    client.km.kernel_spec_manager = ProjectKernelSpecManager()
    client.km.transport_encryption = 'required'
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    print('Executed successfully:', name, flush=True)

if args.mode == 'all':
    for name, digest in script_hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, f'Script/notebook CSV mismatch: {name}'
    (ROOT/'outputs/tables/execution_parity.json').write_text(json.dumps({
        'all_five_scripts_executed': True, 'all_five_notebooks_executed': True,
        'csv_files_byte_identical': len(script_hashes), 'csv_sha256': script_hashes}, indent=2))
    print('All scripts and notebooks produced byte-identical CSVs.', flush=True)
