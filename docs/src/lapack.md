# BLAS and LAPACK reference

```@meta
CurrentModule = StandardPacked
```

`StandardPacked.jl` makes several BLAS and LAPACK functions related to packed matrices available that are not provided by Base.

!!! note "Coverage"
    The general rule of thumb is: Whenever a function is provided in Base for general matrices, this package makes its packed
    counterpart available. Note that there are a lot of additional BLAS and LAPACK functions with respect to packed matrices
    for which no interface is provided because the corresponding general function is not made available in Base.

    If you want to have one of these functions included, just open an issue or pull request.

!!! warning "Naming"
    It is sometimes rather mysterious why a function has the `s` (for symmetric) or `h` (for Hermitian) prefix: for example,
    - [`spr!`](@ref) is the symmetric rank-1 update, and since it is also implemented (though not made available in Base) for
      symmetric complex-valued matrices, there is a separate instance [`hpr!`](@ref) that does the Hermitian equivalent.
    - [`spev!`](@ref) is the symmetric _or_ Hermitian eigensolver (depending on the type), and it does not change its name to
      `hpev!` even though LAPACK does. There is no symmetric complex eigensolver.
    - [`hptrd!`](@ref) is the Hermitian _or_ symmetric tridiagonalized (depending on the type), and it does not change its name
      to `sptrd!` even though LAPACK does.
    This mystery comes from following Base's convention of names.

!!! note "Variants"
    Note that for every function, this package provides an undocumented "raw" variant that does almost no checking apart from
    whether the `uplo` parameter is correct plus a check of the return value. This raw version is usually not made available
    directly by the Julia Base.

    Additionally, `StandardPacked.jl` provide a convenience wrapper that works with the `AbstractArray` interface,
    automatically infers dimensions and checks for the compatibility of all the sizes - this is what is commonly provided by
    Julia. And finally, the vector representing the packed matrix can also be replaced by a [`SPMatrixUnscaled`](@ref) -
    then, the `uplo` argument is not present, as it is determined from the type.

!!! tip "Performance"
    Even the convenience wrappers give more control than you might be used to. Whenever some allocations might occur due to the
    need for temporaries or additional output arrays, these wrappers will allow to pass preallocated arrays if they are sized
    appropriately (where output arguments must match exactly in size, but temporaries may also be larger than necessary).

    Whenever a parameter is documented with a `missing` default, this is such a parameter that might be preallocated if
    necessary. The parameters are all passed by position, not by keyword, which allows Julia to remove unnecessary checks by
    compiling the correct version.

## BLAS Level 2
Some of these functions are already present in `Base.BLAS` and are re-exported in `StandardPacked`. They are listed here for
completeness only.
```@docs
spmv!
hpmv!
spr!
hpr!
```

## BLAS Level 3
```@docs
gemmt!
```

## LAPACK
### Conversion
```@docs
trttp!
tpttr!
```
### Linear systems - Cholesky
```@docs
pptrf!
pptrs!
pptri!
```
### Linear systems - symmetric indefinite
```@docs
spsv!
hpsv!
sptrf!
hptrf!
sptrs!
hptrs!
sptri!
hptri!
```
### Eigenvalue
```@docs
spev!
spevx!
spevd!
spgv!
spgvx!
spgvd!
```
### Tridiagonal
```@docs
hptrd!
opgtr!
opmtr!
```