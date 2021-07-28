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


class DatasetPlotter:
    """
    Wrapper around matplotlib to plot datasets.

    Attributes:
        dataset: The dataset to plot.
        xlim: Interval to plot in x direction.
        ylim: Interval to plot in y direction.
        imgpath: The filesystem path to store images at.
    """

    def __init__(self, dataset: _BaseDataset,
                 xlim: Tuple[float, float] = (None, None),
                 ylim: Tuple[float, float] = (None, None),
                 imgpath: str = None):
        self.dataset = dataset
        self.xlim = xlim
        self.ylim = ylim
        if imgpath is not None:
            pathlib.Path(imgpath).mkdir(parents=True, exist_ok=True)
        self.imgpath = imgpath

    @staticmethod
    def plot(data: np.ndarray,
             labels: np.ndarray,
             classes: List[_BasicClass],
             title: str = "",
             xlim: Tuple[float, float] = (None, None),
             ylim: Tuple[float, float] = (None, None),
             imgpath: str = None):
        """
        Static method to create a dataset plot with color encoded labels.

        The labels indicate the classes each label belongs to.

        Args:
            data: Datapoints (coordinates) to plot.
            labels: The labels assigned to the data.
            classes: The classes, represented by the labels
            title: The title of teh plot.
            xlim: Interval for x coordinates.
            ylim: Interval for y coordinates
            imgpath: Filesystem path to store plot images.
        """
        fig, ax = plt.subplots(dpi=300)
        ax.set_title(title)
        ax.axis(xmin=xlim[0], xmax=xlim[1], ymin=ylim[0], ymax=ylim[1])
        ax.set_xlabel("x1", fontsize='14')
        ax.set_ylabel("x2", fontsize='14')
        ax.tick_params(axis='x', labelsize='14')
        ax.tick_params(axis='y', labelsize='14')

        for circle in classes:
            circle.print(ax=ax)

        # transform labels to colormap
        if labels.shape[1] > 3:
            colors_arr = np.array(list(map(lambda c: colors.rgb2hex(c), plt.get_cmap('tab20').reversed().colors)))
        else:
            colors_arr = MIXABLE_COLORS
        ax.scatter(x=data[:, 0],
                   y=data[:, 1],
                   marker='.',
                   color=DatasetPlotter._labels_to_color(colors_arr, labels))
        pop_r = mlines.Line2D([], [], color=MIXABLE_COLORS[1], label='Red class (1,0)', marker='.', linestyle='None')
        pop_b = mlines.Line2D([], [], color=MIXABLE_COLORS[2], label='Blue class (0,1)', marker='.', linestyle='None')
        pop_p = mlines.Line2D([], [], color=MIXABLE_COLORS[3], label='Both classes (1,1)', marker='.',
                              linestyle='None')

        lgd = ax.legend(handles=[pop_r, pop_b, pop_p],
                        loc='upper left', fontsize=14)
        if imgpath is not None:
            fig.savefig(f"{imgpath}/{Filename.from_title(title)}.png",
                        format="png", bbox_inches='tight')
        return fig

    @staticmethod
    def heatmap(datapoints, temperatures, classes=None, title="", vmax=None, xlim=(None, None), ylim=(None, None), imgpath=None):
        fig, ax = plt.subplots(dpi=300)
        # ax.set_title(title)
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

        if imgpath is not None:
            fig.savefig(f"{imgpath}/{Filename.from_title(title)}.png", format="png", bbox_inches='tight')
        return fig

    @staticmethod
    def confidence_heatmap(datapoints, predictions, classes=None, targets=None, title="", vmax=None, xlim=(None, None), ylim=(None, None), imgpath=None):
        if targets is None:
            conf = np.abs((predictions - 0.5)) * -1 + 0.5
            conf = np.mean(conf, 1)
        else:
            conf = np.abs(predictions - targets)
            conf = np.mean(conf, 1)
        return DatasetPlotter.heatmap(datapoints=datapoints,
                                      temperatures=conf,
                                      classes=classes,
                                      title=title,
                                      vmax=vmax,
                                      xlim=xlim,
                                      ylim=ylim,
                                      imgpath=imgpath)

    def plot_data(self, points: np.ndarray, labels: np.ndarray, title: str = ""):
        """
        Plots given points with color encoded labels.

        Args:
            points: The points to plot.
            labels: The labels for the points.
            title: The title of the plot.

        Returns:
            void
        """
        return DatasetPlotter.plot(data=points,
                                   labels=labels,
                                   title=title,
                                   classes=self.dataset.classes,
                                   xlim=self.xlim,
                                   ylim=self.ylim,
                                   imgpath=self.imgpath)

    def plot_tf_dataset(self, dataset: tf.data.Dataset, title: str = ""):
        """
        Plots a dataset, given as a Tensorflow Dataset API dataset (tf.data.Dataset).

        Args:
            dataset: The dataset to plot.
            title: The title of the plot.

        Returns:
            void
        """
        data, labels = [], []
        for x, y in dataset.unbatch().as_numpy_iterator():
            data.append(x)
            labels.append(y)
        return self.plot_data(np.array(data), np.array(labels), title)

    def plot_confidence(self, points, predictions, targets=None, title="", vmax=0.5):
        return DatasetPlotter.confidence_heatmap(datapoints=points,
                                                 predictions=predictions,
                                                 targets=targets,
                                                 vmax=vmax,
                                                 title=title,
                                                 classes=self.dataset.classes,
                                                 xlim=self.xlim,
                                                 ylim=self.ylim,
                                                 imgpath=self.imgpath)

    def plot_heatmap(self, points, temperatures, title="", vmax=None):
        return DatasetPlotter.heatmap(datapoints=points,
                                      temperatures=temperatures,
                                      vmax=vmax,
                                      title=title,
                                      classes=self.dataset.classes,
                                      xlim=self.xlim,
                                      ylim=self.ylim,
                                      imgpath=self.imgpath)

    @staticmethod
    def _labels_to_color(colors, labels):
        if labels.shape[1] < 5:
            mask = 2 ** np.arange(labels.shape[1])
        else:
            mask = np.zeros((labels.shape[1]), dtype=int)
        c = colors.take((labels.astype(int) * mask).sum(axis=1))
        return c


class HyperparamsPlotter:

    def __init__(self, imgpath: str = None, show_images: bool = False):
        if imgpath is not None:
            pathlib.Path(imgpath).mkdir(parents=True, exist_ok=True)
        self.imgpath = imgpath
        self.show_images = show_images

    def from_gaussian_process(self,
                              df: pd.DataFrame,
                              measure: str,
                              x_col: str,
                              y_col: str,
                              title: str,
                              is_log_scale=True,
                              highlight=None,
                              dataset_size=43645):
        df[x_col + "log"] = np.log10(df[x_col])
        x = df[x_col + "log"]
        df[y_col + "log"] = np.log10(df[y_col])
        y = df[y_col + "log"]

        fig, ax = plt.subplots(nrows=1)
        im = ax.scatter(x=x, y=y, c=df[measure], alpha=1)
        fig.colorbar(im, ax=ax)

        for index, row in df.sort_values(measure, ascending=True).head(20).iterrows():
            ax.plot(row[x_col + "log"], row[y_col + "log"], marker='o', mec='red', mfc="None")

        if highlight is not None:
            ax.plot(np.log10(highlight[1]), np.log10(highlight[0]), marker='o', mec='lime', mfc="None")

        ax.axvline(np.log10(dataset_size), color='grey', linestyle='dashed')
        minrow = df.iloc[df[measure].idxmin()]
        maxrow = df.iloc[df[measure].idxmax()]
        best_vals = {"acc": maxrow, "ece": minrow, "nll": minrow, "f1": maxrow, "map": maxrow, "cost": minrow}

        xlabel = r"$\log_{10}(N)$" if is_log_scale else 'N'
        ylabel = r"$\log_{10}(\tau)$" if is_log_scale else r'$\tau$'
        title = f"{title} \n Measure: {measure}, best: {np.round(best_vals[measure][measure], decimals=4)}, " \
                f"tau: {best_vals[measure]['tau']:.3e}, N={best_vals[measure]['n']:.3e}, "
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if self.imgpath is not None:
            fig.savefig(f"{self.imgpath}/{Filename.from_title(title)}.png", format="png", bbox_inches='tight')
        if self.show_images:
            plt.show()

    def from_gridsearch(self, df: pd.DataFrame, measure: str, x_col: str, y_col: str, title: str, is_log_scale=False):
        df[x_col+"log"] = np.log10(df[x_col])
        x = df[x_col+"log"].unique()
        df[y_col + "log"] = np.log10(df[y_col])
        y = df[y_col + "log"].unique()

        norms = np.arange(y.min(), y.max()+1)
        scales = np.arange(x.min(), x.max()+1)
        c = np.zeros([len(norms), len(scales)])
        for i, norm in enumerate(norms):
            for j, scale in enumerate(scales):
                k = df[(df[x_col+"log"] == float(scale)) & (df[y_col+"log"] == float(norm))]
                c[i, j] = k[measure] if len(k[measure]) == 1 else 1

        minrow = df.iloc[df[measure].idxmin()]
        maxrow = df.iloc[df[measure].idxmax()]
        best_vals = {"acc": maxrow, "ece": minrow, "nll": minrow, "f1": maxrow, "map": maxrow, "cost": minrow}

        fig, ax = plt.subplots(nrows=1)
        im = ax.pcolormesh(scales, norms, c, shading='nearest')
        fig.colorbar(im, ax=ax)

        xstep = x[1] - x[0] if len(x) > 1 else x[0]
        ystep = y[1] - y[0] if len(y) > 1 else y[0]
        for i, (row_index, row) in enumerate(df.sort_values(measure, ascending=True).head(10).iterrows()):
            edgecolor = 'red' if i == 0 else 'pink'
            rect_llc = (row[x_col + "log"] - xstep/2, row[y_col + "log"] - ystep/2)
            rect = patches.Rectangle(rect_llc, xstep, ystep, edgecolor=edgecolor, fc='None')
            ax.add_patch(rect)

        xlabel = r"$\log_{10}(N)$" if is_log_scale else 'N'
        ylabel = r"$\log_{10}(\tau)$" if is_log_scale else r'$\tau$'
        title = f"{title} \n Measure: {measure}, best: {np.round(best_vals[measure][measure], decimals=4)}, " \
                f"tau: {best_vals[measure]['tau']}, N={best_vals[measure]['n']}, "
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if self.imgpath is not None:
            fig.savefig(f"{self.imgpath}/{Filename.from_title(title)}.png", format="png", bbox_inches='tight')
        if self.show_images:
            plt.show()


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
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=300,
                                   gridspec_kw={"height_ratios": [4, 1]}, constrained_layout=True)
        for i, graph in enumerate(spec.graphspecs):
            rd._single_chart(graph.data, with_histogram=with_histogram, ax=ax, legend=spec.legend_for(graph),
                             with_ece=with_ece, figsize=figsize, gidx=i, xlabel=xlabel, ylabel=ylabel)

        fig.show()

    def _single_chart(self,
                      bin_data: Calibration,
                      figsize: Tuple[int, int] = (6, 6),
                      dpi: int = 300,
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

