"""Verify the reference snapshot, script/notebook parity and saved execution outputs."""
from pathlib import Path
import hashlib
import json
import nbformat
from sync_notebooks import NAMES, script_cells

ROOT = Path(__file__).resolve().parents[1]
reference = json.loads((ROOT/'tests/reference_snapshot.json').read_text())
raw_name = 'data/raw/yahoo_ohlcv_20100101_20260916.csv'
same_vintage = hashlib.sha256((ROOT/raw_name).read_bytes()).hexdigest() == reference[raw_name]
if same_vintage:
    for name, digest in reference.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    print('Reference raw snapshot and original Part 1/2 CSVs are byte-identical.')
else:
    print('Different downloaded vintage: exact reference hashes do not apply. Formula checks still apply.')

execution = {}
for script_name, notebook_name in NAMES:
    notebook = nbformat.read(ROOT/'notebooks'/notebook_name, as_version=4)
    expected = script_cells(ROOT/'scripts'/script_name)
    assert [(c.cell_type,c.source) for c in notebook.cells] == [(c.cell_type,c.source) for c in expected]
    cells = [c for c in notebook.cells if c.cell_type == 'code']
    assert [c.execution_count for c in cells] == list(range(1,len(cells)+1))
    for cell in cells:
        for output in cell.outputs:
            assert output.output_type != 'error'
            assert not (output.output_type == 'stream' and output.name == 'stderr'), output
    execution[notebook_name] = dict(executed_code_cells=len(cells), errors=0, warnings=0,
        images=sum('image/png' in o.get('data',{}) for c in cells for o in c.outputs))
    print(notebook_name, execution[notebook_name])

# Earlier files remain local and unchanged; a clean public checkout need not contain backups.
backup = ROOT/'backups/20260919_213942'
if backup.exists():
    for item in json.loads((backup/'manifest.json').read_text()):
        assert hashlib.sha256((backup/item['path']).read_bytes()).hexdigest() == item['sha256']
    print('All five original backup hashes verified.')
parity_path = ROOT/'outputs/tables/execution_parity.json'
assert parity_path.exists(), 'Run the complete pipeline first.'
parity = json.loads(parity_path.read_text())
for name,digest in parity['csv_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
report = dict(reference_vintage=same_vintage, notebooks=execution, csv_parity_verified=True)
(ROOT/'outputs/tables/final_validation.json').write_text(json.dumps(report,indent=2))
print('Final pipeline verification passed.')
