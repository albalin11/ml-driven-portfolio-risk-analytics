"""Validate and commit each stage locally, with hash-checked restart checkpoints."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import nbformat
from sync_notebooks import NAMES, script_cells

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT/'outputs/checkpoints'
MESSAGES = [
    'Part 1: Complete data preparation', 'Part 2: Complete feature engineering',
    'Part 3: Complete volatility forecasting', 'Part 4: Complete VaR and backtesting',
    'Part 5: Complete results and documentation',
]
# Explicit ownership keeps unrelated files and private research data out of commits.
OUTPUTS = {
    1: ['data/raw/yahoo_ohlcv_20100101_20260916.csv', 'data/raw/yahoo_ohlcv_20100101_20260916_metadata.json',
        'data/processed/clean_adjusted_close_prices.csv', 'data/processed/daily_returns.csv',
        'outputs/tables/part1_quality_report.json'],
    2: ['data/processed/volatility_modeling_dataset.csv', 'outputs/tables/feature_summary.csv',
        'outputs/tables/part2_quality_report.json'],
    3: ['outputs/tables/time_splits.csv', 'outputs/tables/validation_candidates.csv',
        'outputs/tables/validation_metrics.csv', 'outputs/tables/validation_predictions.csv',
        'outputs/tables/model_selection.json', 'outputs/tables/test_metrics.csv', 'outputs/tables/test_predictions.csv',
        'outputs/tables/part3_quality_report.json', 'outputs/models/selected_volatility_model.joblib',
        'outputs/figures/test_volatility_forecasts.png'],
    4: ['outputs/tables/risk_forecasts.csv', 'outputs/tables/nonoverlapping_risk_forecasts.csv',
        'outputs/tables/kupiec_backtests.csv', 'outputs/tables/overlapping_violation_summary.csv',
        'outputs/tables/part4_quality_report.json', 'outputs/figures/var_violations.png'],
    5: ['outputs/tables/asset_portfolio_overview.csv', 'outputs/tables/model_comparison.csv',
        'outputs/tables/findings.md', 'outputs/tables/results_summary.json',
        'outputs/figures/asset_portfolio_volatility.png', 'outputs/figures/model_errors.png', 'README.md'],
}
TESTS = {
    1: ['test_data_preparation'],
    2: ['test_feature_engineering'],
    3: ['test_volatility_forecasting'],
    4: ['test_var_backtesting'],
    5: ['test_results_packaging'],
}

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

def stage_files(part):
    script, notebook = NAMES[part-1]
    helpers = {1: ['src/data_preparation_checks.py'],
               2: ['src/feature_engineering.py'],
               4: ['src/var_backtesting.py']}.get(part, [])
    return [f'scripts/{script}', f'notebooks/{notebook}', *helpers, *OUTPUTS[part]]

def file_hashes(names):
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}

def save_state(part, state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR/f'part_{part}.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temporary.replace(path)

def input_hashes(part):
    # Changes to upstream data, formulas, tests or environment invalidate later checkpoints.
    names = ['requirements.txt', '.gitignore', 'tests/test_validation.py', 'tests/test_results_packaging.py',
             'tests/test_volatility_forecasting.py',
             'tests/test_var_backtesting.py',
             'tests/test_checkpoints.py',
             'tests/test_data_preparation.py', 'tests/test_feature_engineering.py', 'tests/reference_snapshot.json',
             'src/run_stages.py', 'src/run_pipeline.py', 'src/sync_notebooks.py', 'src/update_readme.py', 'src/verify_pipeline.py']
    for earlier in range(1, part):
        names += stage_files(earlier)
    return file_hashes(names)

def checkpoint_valid(part, state):
    if state.get('status') != 'complete':
        return False
    try:
        if state['files'] != file_hashes(stage_files(part)) or state['inputs'] != input_hashes(part):
            return False
        return subprocess.run(['git', 'merge-base', '--is-ancestor', state['commit'], 'HEAD'], cwd=ROOT).returncode == 0
    except (FileNotFoundError, KeyError):
        return False

def verify_stage(part):
    script, notebook = NAMES[part-1]
    nb = nbformat.read(ROOT/'notebooks'/notebook, as_version=4)
    assert [(c.cell_type, c.source) for c in nb.cells] == [(c.cell_type, c.source) for c in script_cells(ROOT/'scripts'/script)]
    cells = [c for c in nb.cells if c.cell_type == 'code']
    assert [c.execution_count for c in cells] == list(range(1, len(cells)+1))
    for cell in cells:
        for output in cell.outputs:
            assert output.output_type != 'error'
            assert not (output.output_type == 'stream' and output.name == 'stderr'), output
    file_hashes(stage_files(part))  # Every expected local artifact must exist.
    if part == 1:
        reference = json.loads((ROOT/'tests/reference_snapshot.json').read_text())
        raw = 'data/raw/yahoo_ohlcv_20100101_20260916.csv'
        if file_hashes([raw])[raw] == reference[raw]:
            for name in stage_files(part):
                if name in reference:
                    assert file_hashes([name])[name] == reference[name], name
    if TESTS[part]:
        subprocess.run([sys.executable, '-m', 'unittest', *TESTS[part], '-v'], cwd=ROOT/'tests', check=True)

def public_files(part):
    private = {'outputs/tables/validation_predictions.csv', 'outputs/tables/test_predictions.csv',
               'outputs/tables/risk_forecasts.csv',
               'outputs/tables/nonoverlapping_risk_forecasts.csv'}
    return [name for name in stage_files(part)
            if not name.startswith(('data/', 'outputs/models/')) and name not in private]

def commit_stage(part, names=None, message=None):
    assert not git('diff', '--cached', '--name-only'), 'Existing staged changes must be handled before stage commits.'
    # Verify ignore rules even when credentials or private files do not currently exist.
    protected = ['.venv/probe', '.tools/probe', 'backups/probe', '.env',
                 'outputs/jupyter_runtime/probe', 'outputs/checkpoints/probe',
                 'data/raw/probe.csv', 'data/processed/probe.csv', 'outputs/models/probe.joblib']
    for name in protected:
        assert subprocess.run(['git', 'check-ignore', '-q', name], cwd=ROOT).returncode == 0, name
    names = public_files(part) if names is None else names
    for name in names:
        path = ROOT/name
        assert path.stat().st_size < 10_000_000, name
        if path.suffix != '.png':
            text = path.read_text(encoding='utf-8')
            assert not re.search(r'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN .*PRIVATE KEY-----|C:\\\\Users\\\\|C:\\Users\\', text), name
    subprocess.run(['git', 'add', '--', *names], cwd=ROOT, check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
    if git('diff', '--cached', '--name-only'):
        subprocess.run(['git', 'commit', '-m', message or MESSAGES[part-1]], cwd=ROOT, check=True)
        return git('rev-parse', 'HEAD')
    # Never manufacture an empty or duplicate Part commit.
    return git('log', '-1', '--format=%H', '--', *names)

def run_part(part):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    backup = ROOT/'backups/stage_checkpoints'/f'part_{part}_{stamp}'
    for name in stage_files(part):
        if (ROOT/name).exists():
            destination = backup/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, destination)
    save_state(part, {'status': 'running', 'backup': backup.relative_to(ROOT).as_posix()})
    script, notebook = NAMES[part-1]
    old = nbformat.read(ROOT/'notebooks'/notebook, as_version=4) if (ROOT/'notebooks'/notebook).exists() else None
    nb = nbformat.v4.new_notebook(cells=script_cells(ROOT/'scripts'/script))
    if old:
        nb.metadata = old.metadata
        for index, cell in enumerate(nb.cells):
            if index < len(old.cells):
                cell.id = old.cells[index].id
    nbformat.write(nb, ROOT/'notebooks'/notebook)
    log = STATE_DIR/f'part_{part}_{stamp}.log'
    try:
        with log.open('w', encoding='utf-8') as output:
            subprocess.run([sys.executable, 'src/run_pipeline.py', '--part', str(part)], cwd=ROOT,
                           stdout=output, stderr=subprocess.STDOUT, check=True)
        # Timing metadata alone should not cause a duplicate commit after a rerun.
        new = nbformat.read(ROOT/'notebooks'/notebook, as_version=4)
        if old:
            for index, cell in enumerate(new.cells):
                if index < len(old.cells) and cell.cell_type == 'code' and old.cells[index].cell_type == 'code':
                    previous = old.cells[index]
                    if cell.source == previous.source and cell.outputs == previous.outputs:
                        if 'execution' in previous.metadata:
                            cell.metadata['execution'] = previous.metadata['execution']
            nbformat.write(new, ROOT/'notebooks'/notebook)
        if part == 5:
            subprocess.run([sys.executable, 'src/update_readme.py'], cwd=ROOT, check=True)
        verify_stage(part)
        proof = json.loads((STATE_DIR/f'part_{part}_parity.json').read_text())['csv_sha256']
        csv_parity = {name: proof[name] for name in OUTPUTS[part] if name.endswith('.csv') and name in proof}
        assert csv_parity == file_hashes(list(csv_parity))
        commit = commit_stage(part)
        save_state(part, dict(status='complete', commit=commit, files=file_hashes(stage_files(part)),
                              inputs=input_hashes(part), csv_parity=csv_parity, log=log.relative_to(ROOT).as_posix()))
        print(f'Part {part}: saved and validated; local commit {commit[:7]}.', flush=True)
    except Exception as error:
        save_state(part, dict(status='failed', error=str(error), log=log.relative_to(ROOT).as_posix(),
                              backup=backup.relative_to(ROOT).as_posix()))
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adopt-existing', action='store_true', help='Validate and record already committed complete stages without rerunning them.')
    parser.add_argument('--from-part', type=int, choices=range(1, 6), help='Force rerun this stage and every later stage.')
    args = parser.parse_args()
    assert not (args.adopt_existing and args.from_part), 'Choose adoption or rerunning, not both.'
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if args.adopt_existing:
        names = [name for part in range(1, 6) for name in public_files(part)]
        assert not git('status', '--porcelain', '--', *names), 'Existing stage artifacts must already be committed.'
        subprocess.run([sys.executable, 'src/verify_pipeline.py'], cwd=ROOT, check=True)
        existing_parity = json.loads((ROOT/'outputs/tables/execution_parity.json').read_text())['csv_sha256']
    for part in range(1, 6):
        path = STATE_DIR/f'part_{part}.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        if args.adopt_existing:
            verify_stage(part)
            commit = git('log', '-1', '--format=%H', '--', *public_files(part))
            assert commit, 'No existing stage commit to adopt.'
            csv_parity = {name: existing_parity[name] for name in OUTPUTS[part]
                          if name.endswith('.csv') and name in existing_parity}
            save_state(part, dict(status='complete', commit=commit, files=file_hashes(stage_files(part)),
                                  inputs=input_hashes(part), csv_parity=csv_parity, adopted_existing=True))
            print(f'Part {part}: existing saved results validated; commit {commit[:7]}; no duplicate commit.', flush=True)
        elif (args.from_part is None or part < args.from_part) and checkpoint_valid(part, state):
            print(f'Part {part}: checkpoint verified; reusing commit {state["commit"][:7]}.', flush=True)
        else:
            run_part(part)
    # Refresh the final CSV manifest only after every stage has passed its own checks.
    hashes = {}
    for part in range(1, 6):
        state = json.loads((STATE_DIR/f'part_{part}.json').read_text())
        assert checkpoint_valid(part, state), f'Part {part} changed during this run.'
        hashes.update(state['csv_parity'])
    assert hashes == file_hashes(list(hashes)), 'CSV parity evidence is stale.'
    hashes = dict(sorted(hashes.items()))
    (ROOT/'outputs/tables/execution_parity.json').write_text(json.dumps(dict(
        all_five_scripts_executed=True, all_five_notebooks_executed=True,
        csv_files_byte_identical=len(hashes), csv_sha256=hashes), indent=2))
    subprocess.run([sys.executable, 'src/verify_pipeline.py'], cwd=ROOT, check=True)
    commit_stage(5, names=['outputs/tables/execution_parity.json', 'outputs/tables/final_validation.json'],
                 message='Validate final staged pipeline results')
    print('All stages verified. No push performed; publish only after reviewing the final local commits.', flush=True)

if __name__ == '__main__':
    main()
