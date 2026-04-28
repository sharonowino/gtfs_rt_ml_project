"""
GTFS Disruption Detection - Statistical Significance Testing Module
==================================================================
Statistical tests for comparing model performance.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings

logger = logging.getLogger(__name__)


class SignificanceTester:
    """
    Statistical significance testing for model comparison.
    
    Supports:
    - Paired t-test for cross-validation results
    - Wilcoxon signed-rank test (non-parametric)
    - McNemar's test for classifier comparison
    - Bootstrap confidence intervals
    - Multiple comparison correction (Bonferroni, Holm)
    """
    
    def __init__(self, alpha: float = 0.05, correction: str = 'bonferroni'):
        """
        Initialize significance tester.
        
        Parameters
        ----------
        alpha : float
            Significance level (default: 0.05)
        correction : str
            Multiple comparison correction method
        """
        self.alpha = alpha
        self.correction = correction
    
    def paired_ttest(
        self,
        scores_model1: np.ndarray,
        scores_model2: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict[str, Any]:
        """
        Paired t-test for comparing two models.
        
        Parameters
        ----------
        scores_model1 : np.ndarray
            Performance scores for model 1
        scores_model2 : np.ndarray
            Performance scores for model 2
        alternative : str
            'two-sided', 'less', or 'greater'
        
        Returns
        -------
        Dict with test results
        """
        from scipy import stats
        
        # Compute differences
        differences = scores_model1 - scores_model2
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(scores_model1, scores_model2, alternative=alternative)
        
        # Compute effect size (Cohen's d)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # Compute confidence interval
        n = len(differences)
        se = std_diff / np.sqrt(n)
        ci_lower = mean_diff - stats.t.ppf(1 - self.alpha/2, n-1) * se
        ci_upper = mean_diff + stats.t.ppf(1 - self.alpha/2, n-1) * se
        
        return {
            'test': 'paired_ttest',
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'cohens_d': cohens_d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_samples': n,
            'alternative': alternative
        }
    
    def wilcoxon_test(
        self,
        scores_model1: np.ndarray,
        scores_model2: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict[str, Any]:
        """
        Wilcoxon signed-rank test (non-parametric).
        
        Parameters
        ----------
        scores_model1 : np.ndarray
            Performance scores for model 1
        scores_model2 : np.ndarray
            Performance scores for model 2
        alternative : str
            'two-sided', 'less', or 'greater'
        
        Returns
        -------
        Dict with test results
        """
        from scipy import stats
        
        # Compute differences
        differences = scores_model1 - scores_model2
        
        # Remove zeros
        non_zero = differences[differences != 0]
        
        if len(non_zero) < 10:
            warnings.warn("Fewer than 10 non-zero differences. Results may be unreliable.")
        
        # Perform Wilcoxon test
        statistic, p_value = stats.wilcoxon(non_zero, alternative=alternative)
        
        # Compute effect size (rank-biserial correlation)
        n = len(non_zero)
        mean_rank = n * (n + 1) / 4
        std_rank = np.sqrt(n * (n + 1) * (2*n + 1) / 24)
        z_score = (statistic - mean_rank) / std_rank if std_rank > 0 else 0
        effect_size = 1 - (2 * statistic) / (n * (n + 1)) if n > 0 else 0
        
        return {
            'test': 'wilcoxon',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'z_score': z_score,
            'effect_size': effect_size,
            'n_samples': n,
            'alternative': alternative
        }
    
    def mcnemar_test(
        self,
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray
    ) -> Dict[str, Any]:
        """
        McNemar's test for comparing two classifiers.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred1 : np.ndarray
            Predictions from model 1
        y_pred2 : np.ndarray
            Predictions from model 2
        
        Returns
        -------
        Dict with test results
        """
        from scipy import stats
        
        # Create contingency table
        # b: model1 correct, model2 wrong
        # c: model1 wrong, model2 correct
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)
        
        b = np.sum(correct1 & ~correct2)  # model1 right, model2 wrong
        c = np.sum(~correct1 & correct2)  # model1 wrong, model2 right
        
        # McNemar's test with continuity correction
        if b + c == 0:
            chi2_stat = 0
            p_value = 1.0
        else:
            chi2_stat = (abs(b - c) - 1)**2 / (b + c)
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        
        return {
            'test': 'mcnemar',
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'b': b,  # model1 right, model2 wrong
            'c': c,  # model1 wrong, model2 right
            'n_samples': len(y_true)
        }
    
    def bootstrap_ci(
        self,
        scores: np.ndarray,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Bootstrap confidence interval for a metric.
        
        Parameters
        ----------
        scores : np.ndarray
            Performance scores
        n_bootstrap : int
            Number of bootstrap samples
        ci_level : float
            Confidence level (default: 0.95)
        
        Returns
        -------
        Dict with confidence interval
        """
        # Bootstrap resampling
        bootstrap_means = []
        n = len(scores)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Compute confidence interval
        alpha = 1 - ci_level
        ci_lower = np.percentile(bootstrap_means, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_level': ci_level,
            'n_bootstrap': n_bootstrap
        }
    
    def compare_multiple_models(
        self,
        model_scores: Dict[str, np.ndarray],
        baseline_model: str = None
    ) -> pd.DataFrame:
        """
        Compare multiple models with multiple comparison correction.
        
        Parameters
        ----------
        model_scores : Dict[str, np.ndarray]
            Dictionary mapping model names to performance scores
        baseline_model : str, optional
            Name of baseline model for comparison
        
        Returns
        -------
        pd.DataFrame with comparison results
        """
        from scipy import stats
        
        model_names = list(model_scores.keys())
        
        if baseline_model is None:
            baseline_model = model_names[0]
        
        if baseline_model not in model_scores:
            raise ValueError(f"Baseline model '{baseline_model}' not found")
        
        baseline_scores = model_scores[baseline_model]
        comparisons = []
        
        for model_name in model_names:
            if model_name == baseline_model:
                continue
            
            model_scores_arr = model_scores[model_name]
            
            # Paired t-test
            t_result = self.paired_ttest(baseline_scores, model_scores_arr)
            
            comparisons.append({
                'model': model_name,
                'baseline': baseline_model,
                'mean_baseline': np.mean(baseline_scores),
                'mean_model': np.mean(model_scores_arr),
                'mean_diff': t_result['mean_difference'],
                't_statistic': t_result['t_statistic'],
                'p_value': t_result['p_value'],
                'cohens_d': t_result['cohens_d'],
                'significant': t_result['significant']
            })
        
        # Create DataFrame
        results_df = pd.DataFrame(comparisons)
        
        # Apply multiple comparison correction
        if len(results_df) > 0:
            p_values = results_df['p_value'].values
            
            if self.correction == 'bonferroni':
                corrected_alpha = self.alpha / len(p_values)
                results_df['significant_corrected'] = p_values < corrected_alpha
                results_df['corrected_alpha'] = corrected_alpha
            elif self.correction == 'holm':
                # Holm-Bonferroni correction
                sorted_indices = np.argsort(p_values)
                sorted_p = p_values[sorted_indices]
                n = len(sorted_p)
                
                corrected_alpha = np.zeros(n)
                for i in range(n):
                    corrected_alpha[i] = self.alpha / (n - i)
                
                significant = np.zeros(n, dtype=bool)
                for i in range(n):
                    if sorted_p[i] <= corrected_alpha[i]:
                        significant[i] = True
                    else:
                        break
                
                # Map back to original order
                results_df['significant_corrected'] = False
                for i, idx in enumerate(sorted_indices):
                    results_df.loc[idx, 'significant_corrected'] = significant[i]
                results_df['corrected_alpha'] = self.alpha
            else:
                results_df['significant_corrected'] = results_df['significant']
                results_df['corrected_alpha'] = self.alpha
        
        return results_df
    
    def plot_comparison(
        self,
        model_scores: Dict[str, np.ndarray],
        title: str = 'Model Comparison'
    ):
        """
        Plot model comparison with confidence intervals.
        
        Parameters
        ----------
        model_scores : Dict[str, np.ndarray]
            Dictionary mapping model names to performance scores
        title : str
            Plot title
        """
        import matplotlib.pyplot as plt
        
        model_names = list(model_scores.keys())
        means = [np.mean(scores) for scores in model_scores.values()]
        stds = [np.std(scores) for scores in model_scores.values()]
        
        # Compute confidence intervals
        ci_results = {}
        for name, scores in model_scores.items():
            ci_results[name] = self.bootstrap_ci(scores)
        
        ci_lower = [ci_results[name]['ci_lower'] for name in model_names]
        ci_upper = [ci_results[name]['ci_upper'] for name in model_names]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(model_names))
        ax.bar(x_pos, means, yerr=[np.array(means) - np.array(ci_lower), 
                                     np.array(ci_upper) - np.array(means)],
               capsize=5, alpha=0.7, color='steelblue')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('Performance Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()


def compare_models(
    model_scores: Dict[str, np.ndarray],
    baseline_model: str = None,
    alpha: float = 0.05,
    correction: str = 'bonferroni'
) -> pd.DataFrame:
    """
    Convenience function for model comparison.
    
    Parameters
    ----------
    model_scores : Dict[str, np.ndarray]
        Dictionary mapping model names to performance scores
    baseline_model : str, optional
        Name of baseline model for comparison
    alpha : float
        Significance level
    correction : str
        Multiple comparison correction method
    
    Returns
    -------
    pd.DataFrame with comparison results
    """
    tester = SignificanceTester(alpha=alpha, correction=correction)
    return tester.compare_multiple_models(model_scores, baseline_model)
