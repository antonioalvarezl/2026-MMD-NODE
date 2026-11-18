"""Separate panels of unchanged participant-level mixture moments."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from Analysis.paper_config import ARM_COLORS, ARM_NAMES
from Analysis.paper_style import figsize, NARROW


def save_moment_panels(times, labels, means, covariances, output_dir):
    """Save each coordinate separately using available-case arm means."""
    times, labels = np.asarray(times), np.asarray(labels)
    means, covariances = np.asarray(means), np.asarray(covariances)
    dimension = means.shape[-1]
    if means.shape[:2] != (len(labels), len(times)) or covariances.shape != means.shape + (dimension,):
        raise ValueError("Moment shapes do not match time grid and participants")
    panels = [("global_mean_glucose_by_arm", means[..., 0], r"Mean glucose (mg/dL)"),
              ("global_variance_glucose_by_arm", covariances[..., 0, 0], r"Glucose variance [(mg/dL)$^2$]")]
    if dimension == 2:
        panels += [("global_mean_rate_by_arm", means[..., 1], "Mean glucose rate (mg/dL/min)"),
                   ("global_covariance_glucose_rate_by_arm", covariances[..., 0, 1], r"Glucose-rate covariance [(mg/dL)$^2$/min]"),
                   ("global_variance_rate_by_arm", covariances[..., 1, 1], r"Rate variance [(mg/dL/min)$^2$]")]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for stem, values, ylabel in panels:
        fig, ax = plt.subplots(figsize=figsize(NARROW, 0.72), layout="constrained")
        for arm in (0, 1):
            curves = values[labels == arm]
            ax.plot(times, curves.T, color=ARM_COLORS[arm], alpha=0.12, lw=0.7)
            ax.plot(times, np.nanmean(curves, axis=0), color=ARM_COLORS[arm], lw=2.4, label=ARM_NAMES[arm])
        ax.set(xlabel="Weeks since randomization", ylabel=ylabel, xlim=(times[0], times[-1]))
        ax.grid(True)
        path = output_dir / (stem + ".pdf")
        fig.savefig(path)
        plt.close(fig)
        files.append(path)
    return files
