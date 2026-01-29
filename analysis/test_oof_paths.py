from pathlib import Path
import tempfile
from click.testing import CliRunner
from ced_ml.cli.main import cli
import numpy as np
import pandas as pd

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    results = tmp / 'results'
    results.mkdir()

    # Manually create OOF prediction files where train.py would create them
    lr_dir = results / 'LR_EN' / 'run_test' / 'splits' / 'split_seed42' / 'preds'
    lr_dir.mkdir(parents=True)

    pd.DataFrame({
        'idx': range(100),
        'y_true': np.random.randint(0, 2, 100),
        'y_prob_repeat0': np.random.rand(100),
    }).to_csv(lr_dir / 'train_oof__LR_EN.csv', index=False)

    rf_dir = results / 'RF' / 'run_test' / 'splits' / 'split_seed42' / 'preds'
    rf_dir.mkdir(parents=True)

    pd.DataFrame({
        'idx': range(100),
        'y_true': np.random.randint(0, 2, 100),
        'y_prob_repeat0': np.random.rand(100),
    }).to_csv(rf_dir / 'train_oof__RF.csv', index=False)

    print(f"Created OOF files:")
    print(f"  LR: {lr_dir / 'train_oof__LR_EN.csv'}")
    print(f"  RF: {rf_dir / 'train_oof__RF.csv'}")

    # Try ensemble command
    runner = CliRunner()
    result = runner.invoke(cli, [
        'train-ensemble',
        '--run-id', 'test',
        '--results-dir', str(results),
        '--split-seed', '42',
    ], catch_exceptions=False)

    print(f'\nExit code: {result.exit_code}')
    if result.exit_code != 0:
        print(f'Output:\n{result.output}')
    else:
        print("SUCCESS!")
