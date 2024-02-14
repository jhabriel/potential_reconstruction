"""Module that generates the convergence plot."""

# %% Import modules and convergence plot class
from convergence_plot import ConvergencePlot

input_files = [
    "error_regular_structured",
    "error_irregular_structured",
    "error_unstructured",
    "error_perturbed_unstructured",
]

for input_file in input_files:

    # %% Create an instance of convergence plot
    conv_plot = ConvergencePlot(
        input_file_name=input_file,
        output_file_name=input_file,
        variables_to_plot=[
            ("error_h1_avg", "P1 - Patch Average"),
            ("error_h1_rt0", "P1 - RT0-based"),
            ("error_h1_neu", "P2 - Vohralik"),
        ],
        x_axis_data="mesh_size",
        log_base=2,
    )

    # %% Call the plot method
    conv_plot.plot(
        plot_first_order_line=True,
    )

