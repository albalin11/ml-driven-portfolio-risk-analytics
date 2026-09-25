"""Copy readable percent-format scripts into notebooks without duplicating formulas."""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NAMES = [
    ('part_01_data_preparation.py', '01_data_preparation.ipynb'),
    ('part_02_feature_engineering.py', '02_feature_engineering.ipynb'),
    ('part_03_volatility_forecasting.py', '03_volatility_forecasting.ipynb'),
    ('part_04_var_backtesting.py', '04_var_backtesting.ipynb'),
    ('part_05_results_and_figures.py', '05_results_and_discussion.ipynb'),
]

def script_cells(path):
    cells = []
    kind, lines = None, []
    for line in path.read_text(encoding='utf-8').splitlines() + ['# %%']:
        if line.startswith('# %%'):
            if kind is not None and any(lines):
                if kind == 'markdown':
                    text = '\n'.join(line[2:] if line.startswith('# ') else line[1:] for line in lines).strip()
                    cells.append(nbformat.v4.new_markdown_cell(text))
                else:
                    cells.append(nbformat.v4.new_code_cell('\n'.join(lines).strip()))
            kind = 'markdown' if '[markdown]' in line else 'code'
            lines = []
        else:
            lines.append(line)
    return cells

if __name__ == '__main__':
    for script_name, notebook_name in NAMES:
        path = ROOT / 'notebooks' / notebook_name
        notebook = nbformat.read(path, as_version=4) if path.exists() else nbformat.v4.new_notebook()
        old_cells = notebook.cells
        notebook.cells = script_cells(ROOT / 'scripts' / script_name)
        # Retain execution outputs only for unchanged cells.
        for index, cell in enumerate(notebook.cells):
            if index < len(old_cells) and cell.cell_type == old_cells[index].cell_type:
                cell.id = old_cells[index].id
                if cell.cell_type == 'code' and cell.source == old_cells[index].source:
                    cell.outputs = old_cells[index].outputs
                    cell.execution_count = old_cells[index].execution_count
        notebook.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
        nbformat.write(notebook, path)
        print('Synced:', notebook_name)
