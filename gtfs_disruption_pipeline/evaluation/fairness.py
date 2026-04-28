"""
GTFS Disruption Detection - Fairness Analysis Module
====================================================
Fairness metrics and bias detection for transit disruption models.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings

logger = logging.getLogger(__name__)


class FairnessAnalyzer:
    """
    Fairness analysis for transit disruption prediction models.
    
    Supports:
    - Demographic parity
    - Equalized odds
    - Disparate impact
    - Calibration by group
    - Feature importance by group
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize fairness analyzer.
        
        Parameters
        ----------
        config : Dict, optional
            Configuration for fairness analysis
        """
        self.config = config or {}
        self.results = {}
    
    def demographic_parity(
        self,
        y_pred: np.ndarray,
        protected_attr: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute demographic parity.
        
        Demographic parity requires that the prediction rate is equal
        across protected groups.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Predicted labels
        protected_attr : np.ndarray
            Protected attribute values
        
        Returns
        -------
        Dict with demographic parity metrics
        """
        groups = np.unique(protected_attr)
        group_rates = {}
        
        for group in groups:
            mask = protected_attr == group
            if mask.sum() > 0:
                group_rates[str(group)] = y_pred[mask].mean()
        
        # Compute disparity
        rates = list(group_rates.values())
        if len(rates) > 1:
            max_rate = max(rates)
            min_rate = min(rates)
            disparity = max_rate - min_rate
            ratio = min_rate / max_rate if max_rate > 0 else 1.0
        else:
            disparity = 0.0
            ratio = 1.0
        
        return {
            'group_rates': group_rates,
            'disparity': disparity,
            'ratio': ratio,
            'demographic_parity': disparity < 0.1  # Threshold for fairness
        }
    
    def equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute equalized odds.
        
        Equalized odds requires that TPR and FPR are equal across groups.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        protected_attr : np.ndarray
            Protected attribute values
        
        Returns
        -------
        Dict with equalized odds metrics
        """
        groups = np.unique(protected_attr)
        group_metrics = {}
        
        for group in groups:
            mask = protected_attr == group
            if mask.sum() > 0:
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]
                
                # TPR (True Positive Rate)
                tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
                fn = np.sum((y_true_group == 1) & (y_pred_group == 0))
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                # FPR (False Positive Rate)
                fp = np.sum((y_true_group == 0) & (y_pred_group == 1))
                tn = np.sum((y_true_group == 0) & (y_pred_group == 0))
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                
                group_metrics[str(group)] = {
                    'tpr': tpr,
                    'fpr': fpr,
                    'n_samples': mask.sum()
                }
        
        # Compute disparities
        tprs = [m['tpr'] for m in group_metrics.values()]
        fprs = [m['fpr'] for m in group_metrics.values()]
        
        tpr_disparity = max(tprs) - min(tprs) if len(tprs) > 1 else 0.0
        fpr_disparity = max(fprs) - min(fprs) if len(fprs) > 1 else 0.0
        
        return {
            'group_metrics': group_metrics,
            'tpr_disparity': tpr_disparity,
            'fpr_disparity': fpr_disparity,
            'equalized_odds': tpr_disparity < 0.1 and fpr_disparity < 0.1
        }
    
    def disparate_impact(
        self,
        y_pred: np.ndarray,
        protected_attr: np.ndarray,
        privileged_group: Any = None
    ) -> Dict[str, float]:
        """
        Compute disparate impact ratio.
        
        Disparate impact requires that the selection rate for unprivileged
        groups is at least 80% of the selection rate for privileged groups.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Predicted labels
        protected_attr : np.ndarray
            Protected attribute values
        privileged_group : Any, optional
            Value of privileged group
        
        Returns
        -------
        Dict with disparate impact metrics
        """
        groups = np.unique(protected_attr)
        
        if privileged_group is None:
            privileged_group = groups[0]
        
        if privileged_group not in groups:
            raise ValueError(f"Privileged group {privileged_group} not found in data")
        
        # Compute selection rates
        privileged_mask = protected_attr == privileged_group
        privileged_rate = y_pred[privileged_mask].mean()
        
        unprivileged_groups = [g for g in groups if g != privileged_group]
        unprivileged_rates = {}
        
        for group in unprivileged_groups:
            mask = protected_attr == group
            if mask.sum() > 0:
                unprivileged_rates[str(group)] = y_pred[mask].mean()
        
        # Compute disparate impact ratios
        di_ratios = {}
        for group, rate in unprivileged_rates.items():
            di_ratios[group] = rate / privileged_rate if privileged_rate > 0 else 1.0
        
        # Overall disparate impact
        min_ratio = min(di_ratios.values()) if di_ratios else 1.0
        
        return {
            'privileged_group': str(privileged_group),
            'privileged_rate': privileged_rate,
            'unprivileged_rates': unprivileged_rates,
            'disparate_impact_ratios': di_ratios,
            'min_ratio': min_ratio,
            'disparate_impact': min_ratio >= 0.8  # 80% rule
        }
    
    def calibration_by_group(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        protected_attr: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Compute calibration by group.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_proba : np.ndarray
            Predicted probabilities
        protected_attr : np.ndarray
            Protected attribute values
        n_bins : int
            Number of calibration bins
        
        Returns
        -------
        Dict with calibration metrics by group
        """
        groups = np.unique(protected_attr)
        group_calibration = {}
        
        for group in groups:
            mask = protected_attr == group
            if mask.sum() > 0:
                y_true_group = y_true[mask]
                y_proba_group = y_proba[mask]
                
                # Compute calibration
                bin_edges = np.linspace(0, 1, n_bins + 1)
                bin_indices = np.digitize(y_proba_group, bin_edges) - 1
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)
                
                calibration_data = []
                for bin_idx in range(n_bins):
                    bin_mask = bin_indices == bin_idx
                    if bin_mask.sum() > 0:
                        mean_predicted = y_proba_group[bin_mask].mean()
                        mean_actual = y_true_group[bin_mask].mean()
                        calibration_data.append({
                            'bin': bin_idx,
                            'mean_predicted': mean_predicted,
                            'mean_actual': mean_actual,
                            'n_samples': bin_mask.sum()
                        })
                
                # Compute Expected Calibration Error (ECE)
                ece = 0.0
                for data in calibration_data:
                    ece += data['n_samples'] * abs(data['mean_predicted'] - data['mean_actual'])
                ece /= mask.sum()
                
                group_calibration[str(group)] = {
                    'calibration_data': calibration_data,
                    'ece': ece,
                    'n_samples': mask.sum()
                }
        
        return group_calibration
    
    def analyze_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        protected_attrs: Dict[str, np.ndarray],
        privileged_groups: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive fairness analysis.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_proba : np.ndarray
            Predicted probabilities
        protected_attrs : Dict[str, np.ndarray]
            Dictionary mapping attribute names to values
        privileged_groups : Dict[str, Any], optional
            Dictionary mapping attribute names to privileged group values
        
        Returns
        -------
        Dict with fairness analysis results
        """
        logger.info("="*60)
        logger.info("FAIRNESS ANALYSIS")
        logger.info("="*60)
        
        if privileged_groups is None:
            privileged_groups = {}
        
        results = {}
        
        for attr_name, attr_values in protected_attrs.items():
            logger.info(f"\nAnalyzing fairness for: {attr_name}")
            
            privileged_group = privileged_groups.get(attr_name)
            
            # Demographic parity
            dp = self.demographic_parity(y_pred, attr_values)
            logger.info(f"  Demographic parity disparity: {dp['disparity']:.4f}")
            
            # Equalized odds
            eo = self.equalized_odds(y_true, y_pred, attr_values)
            logger.info(f"  Equalized odds TPR disparity: {eo['tpr_disparity']:.4f}")
            logger.info(f"  Equalized odds FPR disparity: {eo['fpr_disparity']:.4f}")
            
            # Disparate impact
            di = self.disparate_impact(y_pred, attr_values, privileged_group)
            logger.info(f"  Disparate impact min ratio: {di['min_ratio']:.4f}")
            
            # Calibration by group
            calibration = self.calibration_by_group(y_true, y_proba, attr_values)
            
            results[attr_name] = {
                'demographic_parity': dp,
                'equalized_odds': eo,
                'disparate_impact': di,
                'calibration': calibration
            }
        
        self.results = results
        return results
    
    def plot_fairness_metrics(
        self,
        results: Optional[Dict[str, Any]] = None,
        metric: str = 'demographic_parity'
    ):
        """
        Plot fairness metrics.
        
        Parameters
        ----------
        results : Dict[str, Any], optional
            Fairness analysis results
        metric : str
            Metric to plot
        """
        import matplotlib.pyplot as plt
        
        if results is None:
            results = self.results
        
        if not results:
            logger.warning("No fairness results available")
            return
        
        fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
        if len(results) == 1:
            axes = [axes]
        
        for idx, (attr_name, attr_results) in enumerate(results.items()):
            ax = axes[idx]
            
            if metric == 'demographic_parity':
                dp = attr_results['demographic_parity']
                groups = list(dp['group_rates'].keys())
                rates = list(dp['group_rates'].values())
                
                ax.bar(groups, rates, alpha=0.7, color='steelblue')
                ax.axhline(y=np.mean(rates), color='red', linestyle='--', label='Overall rate')
                ax.set_ylabel('Prediction Rate')
                ax.set_title(f'{attr_name}\nDemographic Parity')
                ax.legend()
                
            elif metric == 'equalized_odds':
                eo = attr_results['equalized_odds']
                groups = list(eo['group_metrics'].keys())
                tprs = [m['tpr'] for m in eo['group_metrics'].values()]
                fprs = [m['fpr'] for m in eo['group_metrics'].values()]
                
                x = np.arange(len(groups))
                width = 0.35
                
                ax.bar(x - width/2, tprs, width, label='TPR', alpha=0.7, color='green')
                ax.bar(x + width/2, fprs, width, label='FPR', alpha=0.7, color='red')
                ax.set_xticks(x)
                ax.set_xticklabels(groups)
                ax.set_ylabel('Rate')
                ax.set_title(f'{attr_name}\nEqualized Odds')
                ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def generate_fairness_report(
        self,
        results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate fairness report.
        
        Parameters
        ----------
        results : Dict[str, Any], optional
            Fairness analysis results
        
        Returns
        -------
        str : Fairness report
        """
        if results is None:
            results = self.results
        
        if not results:
            return "No fairness results available"
        
        report = []
        report.append("="*60)
        report.append("FAIRNESS ANALYSIS REPORT")
        report.append("="*60)
        
        for attr_name, attr_results in results.items():
            report.append(f"\nAttribute: {attr_name}")
            report.append("-"*40)
            
            # Demographic parity
            dp = attr_results['demographic_parity']
            report.append(f"Demographic Parity:")
            report.append(f"  Disparity: {dp['disparity']:.4f}")
            report.append(f"  Ratio: {dp['ratio']:.4f}")
            report.append(f"  Fair: {dp['demographic_parity']}")
            
            # Equalized odds
            eo = attr_results['equalized_odds']
            report.append(f"Equalized Odds:")
            report.append(f"  TPR Disparity: {eo['tpr_disparity']:.4f}")
            report.append(f"  FPR Disparity: {eo['fpr_disparity']:.4f}")
            report.append(f"  Fair: {eo['equalized_odds']}")
            
            # Disparate impact
            di = attr_results['disparate_impact']
            report.append(f"Disparate Impact:")
            report.append(f"  Min Ratio: {di['min_ratio']:.4f}")
            report.append(f"  Fair: {di['disparate_impact']}")
        
        return "\n".join(report)


def analyze_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    protected_attrs: Dict[str, np.ndarray],
    privileged_groups: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function for fairness analysis.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray
        Predicted probabilities
    protected_attrs : Dict[str, np.ndarray]
        Dictionary mapping attribute names to values
    privileged_groups : Dict[str, Any], optional
        Dictionary mapping attribute names to privileged group values
    
    Returns
    -------
    Dict with fairness analysis results
    """
    analyzer = FairnessAnalyzer()
    return analyzer.analyze_fairness(y_true, y_pred, y_proba, protected_attrs, privileged_groups)
