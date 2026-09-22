"""Inventory and package the two active PEDAP RCT figure collections."""
from pathlib import Path
import zipfile
from Analysis.paper_config import FIGURES_ROOT, RESULTS_ROOT, PROJECT_ROOT
from Analysis.clinical_support import validate_rct_run

ACTIVE_RUNS = {'d1': 'run14', 'd2': 'run15'}


def write_inventory(figures_root=FIGURES_ROOT, results_root=RESULTS_ROOT):
    root = Path(figures_root)
    paths = []
    for dimension, run in ACTIVE_RUNS.items():
        params = validate_rct_run(Path(results_root) / run)
        if params['d'] != int(dimension[1:]):
            raise ValueError('Run dimension does not match figure collection')
        folder = root / dimension / run
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        paths.extend(sorted(folder.rglob('*.pdf')))
    root.mkdir(parents=True, exist_ok=True)
    (root / 'INVENTORY.txt').write_text(''.join(str(p.relative_to(root)) + '\n' for p in paths))
    return paths


def export_figures(destination=None):
    paths = write_inventory()
    target = Path(destination) if destination else PROJECT_ROOT / 'exports/PEDAP_paper_figures.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(FIGURES_ROOT))
        archive.write(FIGURES_ROOT / 'INVENTORY.txt', 'INVENTORY.txt')
    return target


if __name__ == '__main__':
    print(export_figures())
