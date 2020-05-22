# Predicting polymorphism in molecular crystals using orientational entropy
## Pablo Piaggi and Michele Parrinello
### Proc. Nat. Acad. Sci. 115, 41 (2018)
[![DOI](http://img.shields.io/badge/DOI-10.1073%2Fpnas.1811056115-blue)](https://doi.org/10.1073/pnas.1811056115)
[![arXiv](http://img.shields.io/badge/arXiv-1806.06006-B31B1B.svg)](https://arxiv.org/abs/1806.06006)

This repository contains the input files to reproduce the results of the paper mentioned above. 

The folders Urea and Naphthalene contain an example input for each of these systems.
The source code for the collective variable PairOrientationalEntropy.cpp is compiled on the fly using the LOAD keyword in the plumed.dat file.

I have also included the original source code inside the  ```src``` directory.
If compiling on the fly doesn't work, you need to copy these files into your PLUMED 2 folder and recompile.
Then remove the LOAD line from the plumed.dat files and run the simulations.

Please e-mail me if you have trouble reproducing the results.
