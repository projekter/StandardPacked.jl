# PackedMatrices.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://projekter.github.io/PackedMatrices.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://projekter.github.io/PackedMatrices.jl/dev)
[![Build status (Github Actions)](https://github.com/projekter/PackedMatrices.jl/workflows/CI/badge.svg)](https://github.com/projekter/PackedMatrices.jl/actions)
[![codecov.io](http://codecov.io/github/projekter/PackedMatrices.jl/coverage.svg?branch=main)](http://codecov.io/github/projekter/PackedMatrices.jl?branch=main)

`PackedMatrices` is a Julia package that wraps functions for storing symmetric matrices in their packed form, i.e.,
column-wise stacked upper or lower triangles. It also provides bindings for the corresponding LAPACK functions, which allow
things such as eigendecompositions. Note that typically, the packed form requires a bit longer calculation times and is
slightly more imprecise than the dense form; however, it requires less storage.

Additionally, this package also provides the scaled form of the packed storage, in which off-diagonal elements are scaled by a
factor of $\sqrt2$. This kind of format is very often used as input to semidefinite solvers, as it has the advantage that the
scalar product between the vectorizations of two packed matrices is the same as the Hilbert-Schmidt/Frobenius scalar product
between the actual matrices.

Note that neither of those formats correspond to the rectangular full packed format. The latter has the same favorable scaling
with respect to memory, but additionally a better layout for some linear algebra routines, making it almost as fast as the
dense format. However, index access is a bit weird there and it is not in wide use. Currently, this package does _not_ provide
the RFP format (though it certainly could at some later time). Consider
[`RectangularFullPacked.jl`](https://github.com/JuliaLinearAlgebra/RectangularFullPacked.jl) instead.