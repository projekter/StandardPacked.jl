# Reference

```@meta
CurrentModule = StandardPacked
```

## The SPMatrix type
```@docs
SPMatrix
```

## Creating an SPMatrix
```@docs
SPMatrix(::Integer, ::AbstractVector{R}, ::Symbol) where {R}
SPMatrix{R}(::UndefInitializer, ::Integer, ::Symbol) where {R}
SPMatrix(::Symmetric{R,<:AbstractMatrix{R}}) where {R}
```

## Formats of an SPMatrix
```@docs
SPMatrixUpper
SPMatrixLower
SPMatrixUnscaled
SPMatrixScaled
packed_isupper
packed_islower
packed_isscaled
packed_scale!
packed_unscale!
```

## Accessing the data
```@docs
getindex
setindex!
vec(::SPMatrix)
Matrix{R}(::SPMatrixUnscaled{R}) where {R}
packedsize
packedside
```

## Diagonal and off-diagonal elements
```@docs
PackedDiagonalIterator
rmul_diags!
rmul_offdiags!
lmul_diags!
lmul_offdiags!
```

## Extensions in LinearAlgebra
This section documents functions for which the interface provided by `LinearAlgebra` has been extended to cover the
`SPMatrix` type. However, since `SPMatrix` is an `AbstractMatrix`, many more helper and convenience functions are
actually available that will (hopefully) fall back to the efficient implementations documented here.
```@docs
dot
axpy!
axpby!
factorize
cholesky
cholesky!
bunchkaufman
bunchkaufman!
eigvals
eigvals!
eigmax
eigmax!
eigmin
eigmin!
eigvecs
eigvecs!
eigen
eigen!
diagind
diag
norm
tr
isposdef
isposdef!
transpose
adjoint
checksquare
```

### Low-level matrix operations
```@docs
mul!
lmul!
rmul!
ldiv!
```