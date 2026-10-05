"""Export measured prompt and censored-tail figures; no inferred results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from qwenlean.utils.io import read_jsonl


def main():
    output = Path('reports/figures')
    output.mkdir(exist_ok=True)
    study = json.loads(Path('reports/prompt_policy_measurements.json').read_text())
    rows = read_jsonl('reports/prompt_policy_records.jsonl')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for policy, summary in study['by_policy'].items():
        accuracy = summary['task_mean_accuracy']
        lo, hi = summary['task_accuracy_bootstrap_ci95']
        x = summary['reasoning_tokens']['mean']
        axes[0].errorbar(x, 100*accuracy, yerr=[[100*(accuracy-lo)], [100*(hi-accuracy)]],
                         fmt='o', label=policy, capsize=3)
        axes[0].annotate(policy, (x, 100*accuracy), xytext=(5, 5), textcoords='offset points')
        values = np.sort([r['reasoning_tokens'] for r in rows if r['prompt_policy'] == policy])
        axes[1].step(values, np.arange(1, len(values)+1)/len(values), where='post', label=policy)
    axes[0].set(xlabel='Mean observed reasoning tokens (cap-censored)',
                ylabel='Task-mean accuracy (%)', title='Matched DEV prompt interventions')
    axes[1].set(xlabel='Observed reasoning tokens', ylabel='Empirical cumulative fraction',
                title='Observed lengths, not natural completion lengths')
    axes[1].axvline(study['config']['max_output_tokens'], linestyle=':', color='gray')
    axes[1].legend()
    fig.suptitle('12 tasks × 3 matched seeds per policy; task bootstrap intervals')
    fig.savefig(output/'prompt_compute_accuracy.png', dpi=180)
    plt.close(fig)
    tail_path = Path('reports/termination_tail_measurements.json')
    if tail_path.exists():
        tail = json.loads(tail_path.read_text())
        x = [r['token_horizon'] for r in tail['horizons']]
        lower = [r['survival_lower_bound'] for r in tail['horizons']]
        upper = [r['survival_upper_bound'] for r in tail['horizons']]
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(x, lower, 'o-', label='Known still generating / cohort')
        ax.plot(x, upper, 'o--', label='Including unknown censored outcomes')
        ax.fill_between(x, lower, upper, alpha=.15)
        ax.set(xlabel='Total generated token horizon', ylabel='Fraction of selected cohort',
               ylim=(-.03, 1.03), title='Seven selected DEV trajectories: survival bounds')
        ax.set_xticks(x, [f'{n//1024}K' for n in x])
        ax.legend()
        fig.savefig(output/'termination_survival_bounds.png', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()
