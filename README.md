# A linear potential reconstruction technique for cell-centered finite volume methods

This repository contains the necessary files to reproduce the results from: *A 
linear potential reconstruction technique based on Raviart-Thomas basis functions 
for cell-centered finite volume approximations to the Darcy problem* by J. Varela, C.
E. Schaerer and E. Keilegavlen, submitted to the Proceeding Series of the Brazilian 
Society of Computational and Applied Mathematics, 2024.

## Requirements

To reproduce the results, you will need `Porepy` (commit 10674a7) and `quadpy` 
(version 0.16.10) to be installed in a Python environment.

## Reproducing the results

To reproduce the results, you should run the file `run_analysis.py`. This will 
create several `txt` files inside a (newly created) `errors` folder. 

If you wish to generate the convergence plots, you can now run the file `plot_rates.py`.
This will create several `PDF` files inside a (newly created) `plots` folder.

## Citing

If you use (either all or a part of) the code present in this repository, we ask 
you to cite:

*TODO: Add reference*