# Reference

```@meta
CurrentModule = PackedMatrices
```

## The PackedMatrix type
```@docs
PackedMatrix
```

## Creating a PackedMatrix
```@docs
PackedMatrix(::Integer, ::AbstractVector{R}, ::Symbol) where {R}
PackedMatrix{R}(::UndefInitializer, ::Integer, ::Symbol) where {R}
PackedMatrix(::Symmetric{R,<:AbstractMatrix{R}}) where {R}
```

## Formats of a PackedMatrix
```@docs
PackedMatrixUpper
PackedMatrixLower
PackedMatrixUnscaled
PackedMatrixScaled
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
vec(::PackedMatrix)
Matrix{R}(::PackedMatrixUnscaled{R}) where {R}
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
```@docs
axpy!
mul!
spr!(::Any, ::AbstractVector, ::PackedMatrix)
dot
eigen!
eigvals
eigvals!
eigmin!
eigmax!
cholesky!
isposdef
```