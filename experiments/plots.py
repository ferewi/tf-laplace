import pathlib
import re
from typing import Tuple, List, Union

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import colors, patches
import matplotlib.lines as mlines

from experiments.dataset import _BaseDataset
from experiments.dataset import _BasicClass
from experiments.metrics import Calibration


MIXABLE_COLORS = np.array([
    '#000000',  # black
    '#FF3333',  # red
    '#0198E1',  # blue
    '#BF5FFF',  # purple
    '#FCD116',  # yellow
    '#FF7216',  # orange
    '#4DBD33',  # green
    '#87421F'   # brown
])


class Filename:
    @staticmethod
    def from_title(title):
        filename = re.sub(pattern="[\W\s]+", repl="_", string=title).lower()
        if filename[-1] == "_":
            filename = filename[0:-1]
        return filename


class UncertaintyPlotter:
    """
    Wrapper around matplotlib to plot the model uncertainty.
    """

    @staticmethod
    def heatmap(datapoints, temperatures, classes=None, vmax=None, xlim=(None, None), ylim=(None, None)):
        fig, ax = plt.subplots(dpi=300)
        ax.axis(xmin=xlim[0], xmax=xlim[1], ymin=ylim[0], ymax=ylim[1])
        ax.set_xlabel("x1", fontsize='14')
        ax.set_ylabel("x2", fontsize='14')
        ax.tick_params(axis='x', labelsize='14')
        ax.tick_params(axis='y', labelsize='14')

        hmap = ax.scatter(
            x=datapoints[:, 0],
            y=datapoints[:, 1],
            c=temperatures,
            marker='.',
            cmap=plt.get_cmap('PuRd'),
            vmax=vmax,
            vmin=0.0)
        fig.colorbar(hmap, ax=ax)

        for circle in classes:
            circle.print(ax=ax)

        return fig


class ReliabilityGraphSpec:
    """ Graph Specification data container.

    This class represent the data needed to draw one graph.
    """
    def __init__(self,
                 data: Calibration,
                 netarch: str,
                 nettype: str,
                 classname: str = None):
        self.data = data
        self.netarch = netarch
        self.nettype = nettype
        self.classname = classname

    def __str__(self):
        return f"netarch: {self.netarch} \nnettype: {self.nettype} \nclassname: {self.classname}"


class PlotSpec:
    """ Plot Specification data container

    This class works as a container for all data related to a plot.
    This can be either a single or multiple graphs.
    """
    def __init__(self, graphspecs : Union[List[ReliabilityGraphSpec]] = None, title: str = ""):
        if graphspecs is None:
            graphspecs = []
        self.graphspecs = graphspecs
        self.title_str = title

    def add(self, graphspec: Union[ReliabilityGraphSpec]):
        self.graphspecs.append(graphspec)

    def legend_for(self, graphspec: Union[ReliabilityGraphSpec]):
        num_netarchs = len(dict.fromkeys([gs.netarch for gs in self.graphspecs]))
        num_nettypes = len(dict.fromkeys([gs.nettype for gs in self.graphspecs]))
        if num_netarchs > 1:
            return graphspec.netarch
        if num_nettypes > 1:
            return graphspec.nettype
        return ""


class ReliabilityDiagramPlotter:

    @classmethod
    def from_plot_spec(cls,
                       spec: PlotSpec,
                       with_histogram: bool = False,
                       figsize: Tuple[int, int] = (6, 6),
                       with_ece: bool = True,
                       xlabel: str = "Mean Predicted Value",
                       ylabel: str = "Precision"):
        rd = cls()
        if not with_histogram:
            fig, ax = plt.subplots(figsize=figsize, dpi=72)
            ax = [ax]
        else:
            figsize = (figsize[0], figsize[1] * 1.3)
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=72,
                                   gridspec_kw={"height_ratios": [4, 1]}, constrained_layout=True)
        for i, graph in enumerate(spec.graphspecs):
            rd._single_chart(graph.data, with_histogram=with_histogram, ax=ax, legend=spec.legend_for(graph),
                             with_ece=with_ece, figsize=figsize, gidx=i, xlabel=xlabel, ylabel=ylabel)

        plt.show()

    def _single_chart(self,
                      bin_data: Calibration,
                      figsize: Tuple[int, int] = (6, 6),
                      dpi: int = 72,
                      legend: str = "",
                      with_histogram=True,
                      with_ece: bool = True,
                      ax: List[plt.Axes] = None,
                      xlabel: str = "Mean Predicted Value",
                      ylabel: str = "Precision",
                      gidx: int = 0):
        fig = None
        if ax is None:
            if not with_histogram:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                ax = [ax]
            else:
                figsize = (figsize[0], figsize[0] * 1.3)
                fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=dpi,
                                       gridspec_kw={"height_ratios": [4, 1]})
        main_xlabel = xlabel if not with_histogram else ""
        self._reliability_diagram_subplot(ax[0], bin_data, xlabel=main_xlabel, ylabel=ylabel, gidx=gidx,
                                        legend=legend, with_ece=with_ece)
        if with_histogram:
            hist_y_label = "Counts" if ylabel != "" else ""
            # Draw the confidence histogram upside down.
            orig_counts = bin_data.counts
            bin_data.counts = -bin_data.counts
            self._confidence_histogram_subplot(ax[1], bin_data, title="", xlabel=xlabel, ylabel=hist_y_label, gidx=gidx)
            bin_data.counts = orig_counts

            # Also negate the ticks for the upside-down histogram.
            new_ticks = np.abs(ax[1].get_yticks()).astype(np.int)
            ax[1].set_yticklabels(new_ticks)

            plt.tight_layout()

        if fig is not None:
            return fig

    def _reliability_diagram_subplot(self,
                                     ax: plt.Axes,
                                     bin_data: Calibration,
                                     legend='',
                                     xlabel="Mean Predicted Value",
                                     ylabel="Precision",
                                     with_ece=True,
                                     gidx=0):
        """Draws a reliability diagram into a subplot."""
        precisions = bin_data.precisions
        counts = bin_data.counts
        bins = bin_data.bins

        bin_size = 1.0 / len(counts)
        positions = bins[:-1] + bin_size / 2.0

        if with_ece:
            ece = (bin_data.ece * 100)
            legend = f"{legend}, ECE={ece:.2f}%"

        acc_plt = ax.errorbar(positions + gidx * (0.01), precisions, marker="o", label=legend, yerr=bin_data.acc_erors)
        ax.legend(fontsize='12')

        ax.set_aspect("equal")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', labelsize='14')
        ax.tick_params(axis='y', labelsize='14')
        ax.set_xlabel(xlabel, fontsize='14')
        ax.set_ylabel(ylabel, fontsize='14')

    def _confidence_histogram_subplot(self,
                                      ax: plt.Axes,
                                      bin_data: Calibration,
                                      title="Examples per bin",
                                      xlabel="Mean Predicted Value",
                                      ylabel="Count",
                                      gidx=0):
        """Draws a confidence histogram into a subplot."""
        counts = bin_data.counts
        bins = bin_data.bins

        bin_size = 1.0 / len(counts)
        width = bin_size * 0.4
        positions = bins[:-1] + bin_size / 2.0 + gidx*width
        ax.bar(positions, counts, width=width, yerr=bin_data.counts_errors)

        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.tick_params(axis='x', labelsize='14')
        ax.tick_params(axis='y', labelsize='14')
        ax.set_xlabel(xlabel, fontsize='14')
        ax.set_ylabel(ylabel, fontsize='14')


def roc_plot(roc_base, roc_bnn):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('ROC of in- vs. out-of-distribution for F3 Dataset')
    ax.plot(roc_base['fpr'], roc_base['tpr'], label=f"baseline, AUC = {roc_base['auroc']:.4f}")
    # ax.plot(fpr_conf, tpr_conf, label=f"bnn conf, AUC = {auroc_conf:.4f}")
    ax.plot(roc_bnn['fpr'], roc_bnn['tpr'], label=f"stddev, AUC = {roc_bnn['auroc']:.4f}")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.show()