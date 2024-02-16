"""Module that generates the convergence plots."""

# %% Import modules and convergence plot class
from convergence_plot import ConvergencePlot

input_files = [
    "error_regular_structured",
    "error_irregular_structured",
    "error_unstructured",
    "error_perturbed_unstructured",
]

for input_file in input_files:

    # L2 convergence
    l2_conv_plot = ConvergencePlot(
        input_file_name=input_file,
        output_file_name=input_file + "_l2",
        variables_to_plot=[
            ("error_l2_p0", "P0"),
            ("error_l2_avg", "P1 - Patch Average"),
            ("error_l2_rt0", "P1 - RT0-based"),
            ("error_l2_neu", "P2 - Vohralik"),
        ],
        x_axis_data="mesh_size",
        log_base=2,
    )
    l2_conv_plot.plot(
        plot_second_order_line=True,
        plot_first_order_line=True,
    )

    # H1 convergence
    h1_conv_plot = ConvergencePlot(
        input_file_name=input_file,
        output_file_name=input_file + "_h1",
        variables_to_plot=[
            ("error_h1_avg", "P1 - Patch Average"),
            ("error_h1_rt0", "P1 - RT0-based"),
            ("error_h1_neu", "P2 - Vohralik"),
        ],
        x_axis_data="mesh_size",
        log_base=2,
    )
    h1_conv_plot.plot(
        plot_first_order_line=True,
    )

