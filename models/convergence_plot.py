"""Module containing functionality for plotting convergence analysis."""
from __future__ import annotations

import numpy as np
import matplotlib.figure
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import warnings

from typing import Literal, Optional


import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rc("font", family="Times New Roman")
plt.rcParams.update({"text.usetex": True})
plt.rc("text.latex", preamble=r"\usepackage{bm}")


class ConvergencePlot:
    """Plot convergence rate in space or time with legends."""

    def __init__(
        self,
        input_file_name: str,
        input_format: str = "txt",
        output_file_name: str = "images/convergence_plot",
        output_format: str = "pdf",
        variables_to_plot: Optional[list[tuple[str, str]]] = None,
        x_axis_data: Literal["mesh_size", "time_step"] = "mesh_size",
        log_base: Optional[int] = None,
    ):
        # Input file

        self.input_file: str = input_file_name + "." + input_format

        # Output file
        self.output_file: str = output_file_name + "." + output_format

        # Read input file and retrieve data to plot
        input_data: list[tuple[str, np.ndarray]] = self._read_input_file()

        # Retrieve mesh sizes, time steps, and all errors present the input data
        self.mesh_sizes: np.ndarray = input_data[0][1]
        # self.time_steps: np.ndarray = input_data[1][1]
        self._all_errors: list[tuple[str, np.ndarray]] = input_data[1:]

        # Get x-axis data
        self._x_axis_data: str = x_axis_data

        # Set x_axis_values
        # Note that the "true" x-data plotted is log_n(1 / x_axis_data)
        if self._x_axis_data == "mesh_size":
            self._x_values: np.ndarray = self.mesh_sizes
        else:
            self._x_values = self.time_steps

        # Define base of logarithm
        if log_base is not None:
            self._log_base: int = log_base
        else:
            self._log_base = np.int8(self._x_values[0] / self._x_values[1])

        # Get variables to plot
        self._variables_to_plot = variables_to_plot

        # Now, we filter variables that should be plotted based on `variables_to_plot`.
        # Each tuple from the list has the form: (variable_name, label, values).
        self.errors_to_plot: list[
            tuple[str, str, np.ndarray]
        ] = self._get_errors_to_plot()

    # -----> Main plot method
    def plot(
        self,
        plot_first_order_line=False,
        plot_second_order_line=False,
        show_plot=False,
    ) -> None:
        """Plot convergence rates and legends.

        Parameters:
            plot_first_order_line: ...
            plot_second_order_line: ...
            show_plot: ...

        """
        # Create the figure
        fig, ax = self._create_figure()

        # Define the colormap. tab10 by default.
        cmap = self._define_color_palette()

        # Plot first order convergence line (might require tweaking)
        if plot_first_order_line:
            self._first_order_line(ax)

        # Plot second order convergence line (might require tweaking)
        if plot_second_order_line:
            self._second_order_line(ax)

        # Set x-label and y-label
        self._set_axis_labels(ax)

        # Loop through the variables to plot
        for idx, var in enumerate(self.errors_to_plot):
            self._plot_rate(ax, cmap, idx, var)
            self._add_legend(ax, cmap, idx, var)

        # Plot legend (might require tweaking)
        self._legend(ax)

        # Extra plot configurations
        self._extra_plot_settings(ax, show_plot)

        # Export image
        plt.savefig(self.output_file)

    # -----> Helper functions

    def _create_figure(self) -> tuple[matplotlib.figure.Figure, np.ndarray]:
        """Create the matplotlib figure.

        Returns:
            Tuple with two elements:
                - ...
                - ...

        """
        fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 2]})
        return fig, ax

    def _define_color_palette(self) -> mcolors.ListedColormap:
        """Define the color palette to be used for plotting the different rates.

        Returns:
            ...

        """
        # Use tab20 if you have more than 10 variables to plot
        # You can also declare a custom palette if you want
        return mcolors.ListedColormap(plt.cm.tab10.colors[: len(self.errors_to_plot)])

    def _read_input_file(self) -> list[tuple[str, np.ndarray]]:
        """Read the input file.

        Returns:
            List of 2-tuples of length ``2 + num_variables_to_plot``. The first item
            contains the mesh sizes, the second item contains the time step sizes,
            and the rest contain the errors.

        """
        # Open file and retrieve the header
        with open(self.input_file) as f:
            lines = f.readlines()
        header = lines[0]

        # Strip '#' from header
        header = header.lstrip("# ")

        # Strip '\n' from header
        header = header.rstrip("\n")

        # Get all variable names
        names = header.split()

        # Load the file
        values = np.loadtxt(fname=self.input_file, dtype=float, unpack=True)

        # Prepare to return
        input_data: list[tuple[str, np.ndarray]] = [
            (name, val) for (name, val) in zip(names, values)
        ]

        return input_data

    def _get_errors_to_plot(self) -> list[tuple[str, str, np.ndarray]]:
        """Filter errors to be plotted.

        Returns:
            ...

        """
        errors_to_plot: list[tuple[str, str, np.ndarray]] = []
        if self._variables_to_plot is None:
            for idx in range(len(self._all_errors)):
                errors_to_plot.append(
                    (
                        self._all_errors[idx][0],  # variable name
                        self._all_errors[idx][0],  # label
                        self._all_errors[idx][1],  # values
                    )
                )
        else:
            var_names: list[str] = [var[0] for var in self._all_errors]
            for name, label in self._variables_to_plot:
                if name in var_names:
                    idx: int = var_names.index(f"{name}")
                    errors_to_plot.append(
                        (
                            name,  # variable name
                            label,  # label
                            self._all_errors[idx][1],  # values
                        )
                    )
                else:
                    warnings.warn(message=f"{name} not present in {self.input_file}.")

        return errors_to_plot

    def _first_order_line(self, axes: np.ndarray) -> None:
        """Plot first order line.

        Parameters:
            axes:

        """
        x0 = self._log_n(1 / self._x_values[0])
        x1 = self._log_n(1 / self._x_values[-1])
        y0 = -1  # -9 for parabolic and -1 for trigonometric
        y1 = y0 - (x1 - x0)
        # Plot line
        axes[0].plot(
            [x0, x1],
            [y0, y1],
            linewidth=3,
            linestyle="-",
            color="black",
        )
        # Ghost plot for legend
        axes[1].plot(
            [],
            [],
            linewidth=3,
            linestyle="-",
            color="black",
            label="First order",
        )

    def _second_order_line(self, axes: np.ndarray) -> None:
        """Plot second order line.

        Parameters:
            axes:

        """
        x0 = self._log_n(1 / self._x_values[0])
        x1 = self._log_n(1 / self._x_values[-1])
        y0 = -1  # Modify this value according to your plot
        y1 = y0 - 2 * (x1 - x0)
        # Plot line
        axes[0].plot(
            [x0, x1],
            [y0, y1],
            linewidth=3,
            linestyle="--",
            color="black",
        )
        # Ghost plot for legend
        axes[1].plot(
            [],
            [],
            linewidth=3,
            linestyle="--",
            color="black",
            label="Order 2",
        )

    def _log_n(self, array: np.ndarray) -> np.ndarray:
        return np.emath.logn(self._log_base, array)

    def _set_axis_labels(self, axes) -> None:
        """Set labels of the `x` and `y` axis.

        Parameters:
            axes: ...

        """
        # Set x-label
        if self._x_axis_data == "mesh_size":
            axes[0].set_xlabel(rf"$\log_{self._log_base}$($1/h$)", fontsize=16)
        else:
            axes[0].set_xlabel(rf"$\log_{self._log_base}$($1/\tau$)", fontsize=16)
        # Set y-label
        axes[0].set_ylabel(rf"$\log_{self._log_base}(\varepsilon)$", fontsize=16)

    def _plot_rate(
        self,
        axes: np.ndarray,
        colormap: mcolors.ListedColormap,
        index: int,
        variable: tuple[str, str, np.ndarray],
    ) -> None:
        """Plot the line corresponding to the given `variable`.

        Parameters:
            axes: ...
            colormap: ...
            index: ...
            variable: ...

        """
        axes[0].plot(
            self._log_n(1 / self._x_values),
            self._log_n(variable[2]),
            linestyle="-",
            linewidth=2,
            marker="o",
            markersize=5,
            color=colormap.colors[index],
        )
        axes[0].tick_params(axis="both", which="major", labelsize=13)

    def _add_legend(
        self,
        axes: np.ndarray,
        colormap: mcolors.ListedColormap,
        index: int,
        variable: tuple[str, str, np.ndarray],
    ) -> None:
        """Plot the legend corresponding to the given `variable.

        Parameter:
            axes: ...
            colormap: ...
            index: ...
            variable: ...

        """
        axes[1].plot(
            [],
            [],
            linestyle="-",
            linewidth=3,
            marker="o",
            markersize=6,
            color=colormap.colors[index],
            label=f"{variable[1]}",
        )

    def _legend(self, axes) -> None:
        """Include the legend in the right subplot."""
        axes[1].legend(
            bbox_to_anchor=(1.05, 0.5),  # this most-likely requires tweaking
            loc="center right",
            fontsize=16,
        )
        axes[1].axis("off")

    def _extra_plot_settings(self, axes: np.ndarray, show_plot: bool) -> None:
        """Extra settings of the plot."""
        plt.tight_layout()
        # Add here more settings if needed ...
        if show_plot:
            plt.show()