import pandas as pd
from pathlib import Path
import glob

results_dir = Path(__file__).parent

# Load all summaries
oof_files = sorted(glob.glob(str(results_dir / "summary_oof_seed*.csv")))
test_files = sorted(glob.glob(str(results_dir / "summary_test_seed*.csv")))

print(f"Found {len(oof_files)} OOF + {len(test_files)} test summaries")

all_data = []
for f in oof_files + test_files:
    try:
        df = pd.read_csv(f)
        all_data.append(df)
    except Exception as e:
        print(f"WARNING: Could not load {f}: {e}")

if not all_data:
    print("ERROR: No data to aggregate")
    exit(1)

combined = pd.concat(all_data, ignore_index=True)

# Per-model summary
print("\n" + "="*80)
print("FULL CASE COVERAGE SUMMARY")
print("="*80)

for model in sorted(combined['model'].unique()):
    model_data = combined[combined['model'] == model]
    oof_data = model_data[model_data['mode'] == 'oof']
    test_data = model_data[model_data['mode'] == 'test']

    print(f"\n{model}:")
    print(f"  OOF runs:  {len(oof_data)} (~{oof_data['n_incident'].sum()} incident instances)")
    print(f"  Test runs: {len(test_data)} (~{test_data['n_incident'].sum()} incident instances)")

    if 'median_diff' in model_data.columns:
        avg_diff = model_data['median_diff'].mean()
        sig_count = sum(model_data['mw_pval'] < 0.05)
        print(f"  Avg median difference: {avg_diff:+.4f}")
        print(f"  Significant runs (p<0.05): {sig_count}/{len(model_data)}")
        print(f"  Avg Cohen's d: {model_data['cohens_d'].mean():.3f}")

# Save consolidated
output_path = results_dir / "FULL_COVERAGE_SUMMARY.csv"
combined.to_csv(output_path, index=False)
print(f"\nFull coverage summary: {output_path.name}")

# Per-model aggregate
per_model = combined.groupby(['model']).agg({
    'split_seed': 'count',
    'n_incident': 'sum',
    'n_prevalent': 'sum',
    'median_diff': ['mean', 'std'],
    'cohens_d': ['mean', 'std'],
    'mw_pval': lambda x: sum(x < 0.05)
}).round(4)

model_summary_path = results_dir / "FULL_COVERAGE_MODEL_SUMMARY.csv"
per_model.to_csv(model_summary_path)
print(f"Per-model summary: {model_summary_path.name}")
