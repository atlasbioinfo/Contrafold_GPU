# NOTICE

This software is a GPU reimplementation of the **CONTRAfold** RNA secondary
structure model and bundles its trained parameter file
(`data/contrafold.params.complementary`).

CONTRAfold:
> Do CB, Woods DA, Batzoglou S. "CONTRAfold: RNA secondary structure prediction
> without physics-based models." Bioinformatics. 2006 Jul 15;22(14):e90-8.
> http://contra.stanford.edu/contrafold/

CONTRAfold is distributed under the BSD license. The scoring functions, parameter
semantics, and inside recurrence implemented here are derived from the CONTRAfold
source code; the parameter file is taken verbatim from the CONTRAfold distribution.

Please retain this attribution and cite the CONTRAfold paper when using this package.
The GPU reimplementation is provided for research use.
