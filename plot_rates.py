"""Module that generates the convergence plots."""
import os
from convergence_plot import ConvergencePlot

input_files = [
    "error_regular_structured_para",
    "error_irregular_structured_para",
    "error_unstructured_para",
    "error_perturbed_unstructured_para",
    "error_regular_structured_trig",
    "error_irregular_structured_trig",
    "error_unstructured_trig",
    "error_perturbed_unstructured_trig",
]

# Create plots folder if it does not exist
if not os.path.exists("plots"):
    os.makedirs("plots")

for input_file in input_files:

    # ----> Uncomment these lines to plot L2 convergence rates
    # # L2 convergence
    # l2_conv_plot = ConvergencePlot(
    #     input_file_name="errors/" + input_file,
    #     output_file_name="plots/" + input_file + "_l2",
    #     variables_to_plot=[
    #         ("error_l2_p0", "P0"),
    #         ("error_l2_rt0", "P1-RT0-BASED"),
    #         ("error_l2_avg", "P1-CC-AVG"),
    #         ("error_l2_neu", "P2-RECON"),
    #     ],
    #     x_axis_data="mesh_size",
    #     log_base=2,
    # )
    # l2_conv_plot.plot(
    #     plot_second_order_line=True,
    #     plot_first_order_line=True,
    # )

    # H1 convergence
    h1_conv_plot = ConvergencePlot(
        input_file_name="errors/" + input_file,
        output_file_name="plots/" + input_file + "_h1",
        variables_to_plot=[
            ("error_h1_rt0", "P1-RT0-BASED"),
            ("error_h1_avg", "P1-CC-AVG"),
            ("error_h1_neu", "P2-RECON"),
        ],
        x_axis_data="mesh_size",
        log_base=2,
    )
    h1_conv_plot.plot(
        plot_first_order_line=True,
    )

