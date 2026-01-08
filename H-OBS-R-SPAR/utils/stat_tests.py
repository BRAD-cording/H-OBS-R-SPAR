"""
Statistical Tests for Experiment Validation

Implements t-test, Bootstrap, and confidence interval calculations
for validating experimental results per T-SMC-S requirements.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


def paired_t_test(
    baseline: np.ndarray,
    method: np.ndarray,
    alpha: float = 0.05
) -> Dict:
    """
    Perform paired t-test between baseline and method results.
    
    Args:
        baseline: Baseline accuracy/metric values
        method: Proposed method values
        alpha: Significance level
    
    Returns:
        Dictionary with t-statistic, p-value, and significance
    """
    t_stat, p_value = stats.ttest_rel(baseline, method)
    significant = p_value < alpha
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'alpha': alpha,
        'significant': significant,
        'interpretation': 'Significant difference' if significant else 'No significant difference'
    }


def bootstrap_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.
    
    Args:
        data: Sample data
        confidence: Confidence level (e.g., 0.95 for 95%)
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        (lower_bound, upper_bound)
    """
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
    
    return lower, upper


def compute_effect_size(baseline: np.ndarray, method: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    pooled_std = np.sqrt((np.var(baseline) + np.var(method)) / 2)
    if pooled_std == 0:
        return 0.0
    return (np.mean(method) - np.mean(baseline)) / pooled_std


def run_statistical_analysis(
    results: Dict[str, np.ndarray],
    baseline_name: str = 'Unpruned'
) -> Dict:
    """
    Run complete statistical analysis comparing all methods to baseline.
    
    Args:
        results: Dictionary mapping method names to result arrays
        baseline_name: Name of baseline method
    
    Returns:
        Dictionary with all statistical results
    """
    baseline = results[baseline_name]
    analysis = {}
    
    for method_name, method_results in results.items():
        if method_name == baseline_name:
            continue
        
        # T-test
        ttest = paired_t_test(baseline, method_results)
        
        # Bootstrap CI
        ci_lower, ci_upper = bootstrap_confidence_interval(method_results)
        
        # Effect size
        effect = compute_effect_size(baseline, method_results)
        
        analysis[method_name] = {
            'mean': np.mean(method_results),
            'std': np.std(method_results),
            'ci_95': (ci_lower, ci_upper),
            't_test': ttest,
            'effect_size': effect
        }
    
    return analysis


if __name__ == "__main__":
    # Test with example data
    np.random.seed(42)
    
    results = {
        'Unpruned': np.array([76.13, 76.15, 76.12, 76.14, 76.11]),
        'H-OBS/R-SPAR': np.array([75.97, 75.95, 75.96, 75.98, 75.94]),
        'DepGraph': np.array([75.82, 75.80, 75.85, 75.81, 75.84]),
    }
    
    analysis = run_statistical_analysis(results)
    
    print("=== Statistical Analysis ===\n")
    for method, stats_data in analysis.items():
        print(f"{method}:")
        print(f"  Mean ± Std: {stats_data['mean']:.2f} ± {stats_data['std']:.2f}")
        print(f"  95% CI: [{stats_data['ci_95'][0]:.2f}, {stats_data['ci_95'][1]:.2f}]")
        print(f"  p-value: {stats_data['t_test']['p_value']:.4f}")
        print(f"  Effect size: {stats_data['effect_size']:.4f}")
        print()
