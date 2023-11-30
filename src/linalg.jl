export eigmin!, eigmax!, eigvecs!

const MaybeComplex{R} = Union{R,Complex{R}}

"""
    dot(::SPMatrix, ::SPMatrix)

Gives the scalar product between two packed matrices, which corresponds to the Frobenius/Hilbert-Schmidt scalar product for
matrices. This is fastest for scaled matrices.
"""
function LinearAlgebra.dot(A::SPMatrix{R1,V,Fmt} where {V}, B::SPMatrix{R2,V,Fmt} where {V}) where {R1,R2,Fmt}
    A.dim == B.dim || error("Matrices must have same dimensions")
    if packed_isscaled(Fmt)
        return dot(A.data, B.data)
    else
        result = zero(promote_type(R1, R2))
        diags = SPDiagonalIterator(A)
        cur_diag = iterate(diags)
        i = 1
        @inbounds while !isnothing(cur_diag)
            @simd for j in i:cur_diag[1]-1
                result += 2conj(A[j]) * B[j]
            end
            result += conj(A[cur_diag[1]]) * B[cur_diag[1]]
            i = cur_diag[1] +1
            cur_diag = iterate(diags, cur_diag[2])
        end
        return result
    end
end
function LinearAlgebra.dot(A::SPMatrix{R1,<:SparseVector,Fmt}, B::SPMatrix{R2,V,Fmt} where {V}) where {R1,R2,Fmt}
    A.dim == B.dim || error("Matrices must have same dimensions")
    if packed_isscaled(Fmt)
        return dot(A.data, B.data)
    else
        result = zero(promote_type(R1, R2))
        nzs = rowvals(A.data)
        vs = nonzeros(A.data)
        diags = SPDiagonalIterator(A)
        cur_diag = iterate(diags)
        isnothing(cur_diag) && return result
        @inbounds for i in 1:length(nzs) # for @simd, cannot iterate over (j, v)
            j = nzs[i]
            v = vs[i]
            while cur_diag[1] < j
                cur_diag = iterate(diags, cur_diag[2])
                if isnothing(cur_diag)
                    cur_diag = (typemax(Int), cur_diag[2])
                end
            end
            if cur_diag[1] == j
                result += conj(v) * B[j]
            else
                result += 2conj(v) * B[j]
            end
        end
        return result
    end
end
LinearAlgebra.dot(A::SPMatrix{R}, B::SPMatrix{R,<:SparseVector}) where {R} = conj(dot(B, A))
function LinearAlgebra.dot(A::SPMatrix{R1,<:SparseVector,Fmt}, B::SPMatrix{R2,<:SparseVector,Fmt}) where {R1,R2,Fmt}
    A.dim == B.dim || error("Matrices must have same dimensions")
    if packed_isscaled(Fmt)
        return dot(A.data, B.data)
    else
        result = zero(promote_type(R1, R2))
        nzA = rowvals(A.data)
        nzB = rowvals(B.data)
        vA = nonzeros(A.data)
        vB = nonzeros(B.data)
        iAmax = length(nzA)
        iBmax = length(nzB)
        iA = 1
        iB = 1
        diags = SPDiagonalIterator(A)
        cur_diag = iterate(diags)
        isnothing(cur_diag) && return result
        @inbounds while iA ≤ iAmax && iB ≤ iBmax
            pA = nzA[iA]
            pB = nzB[iB]
            while pA > pB
                iB += 1
                iB > iBmax && return result
                pB = nzB[iB]
            end
            while pB > pA
                iA += 1
                iA > iAmax && return result
                pA = nzA[iA]
            end
            @assert pA == pB
            while cur_diag[1] < pA
                cur_diag = iterate(diags, cur_diag[2])
                if isnothing(cur_diag)
                    cur_diag = (typemax(Int), cur_diag[2])
                end
            end
            if cur_diag[1] == pA
                result += conj(vA[iA]) * vB[iB]
            else
                result += 2conj(vA[iA]) * vB[iB]
            end
            iA += 1
            iB += 1
        end
        return result
    end
end

"""
    axpy!(α, x::Union{<:SPMatrix,<:AbstractVector},
        y::Union{<:SPMatrix,<:AbstractVector})

Overwrite `y` with `x * α + y` and return `y`, where both `x` and `y` are interpreted as their corresponding vectorizations.
"""
LinearAlgebra.axpy!(α, x::SPMatrix, y::AbstractVector) = LinearAlgebra.axpy!(α, x.data, y)
LinearAlgebra.axpy!(α, x::AbstractVector, y::SPMatrix) = LinearAlgebra.axpy!(α, x, y.data)
LinearAlgebra.axpy!(α, x::SPMatrix, y::SPMatrix) = LinearAlgebra.axpy!(α, x.data, y.data)

"""
    axpby!(α, x::Union{<:SPMatrix,<:AbstractVector}, β,
        y::Union{<:SPMatrix,<:AbstractVector})

Overwrite `y` with `x * α + y * β` and return `y`, where both `x` and `y` are interpreted as their corresponding
vectorizations.
"""
LinearAlgebra.axpby!(α, x::SPMatrix, β, y::AbstractVector) = LinearAlgebra.axpby!(α, x.data, β, y)
LinearAlgebra.axpby!(α, x::AbstractVector, β, y::SPMatrix) = LinearAlgebra.axpby!(α, x, β, y.data)
LinearAlgebra.axpby!(α, x::SPMatrix, β, y::SPMatrix) = LinearAlgebra.axpby!(α, x.data, β, y.data)

"""
    factorize(A::SPMatrix)

Compute a Bunch-Kaufman factorization of `A`.
"""
LinearAlgebra.factorize(P::SPMatrix{<:Real}) = bunchkaufman(P)

# construction of various types such as Diagonal automatically works by implementing diag.
# Bidiagonal(::SPMatrix, ::Symbol) also works. We don't infer the symbol from the SPMatrix type, as in the hermitian
# case, it's up to the user to decide if the super- or subdiagonal is requested.


"""
    cholesky(P::SPMatrix, NoPivot(); shift=0, check=true) -> Cholesky

Compute the Cholesky factorization of a packed positive definite matrix `P` and return a `Cholesky` factorization.

The triangular Cholesky factor can be obtained from the factorization `F` via `F.L` and `F.U`, where
`P ≈ F.U' * F.U ≈ F.L * F.L'`.

The following functions are available for Cholesky objects from packed matrices: `size`, `\\`, `inv`, `det`, `logdet`,
`isposdef`.

When `check = true`, an error is thrown if the decomposition fails. When `check = false`, responsibility for checking the
decomposition's validity (via issuccess) lies with the user.
"""
LinearAlgebra.cholesky(P::SPMatrix{T}, ::NoPivot=NoPivot(); shift::R=zero(R),
    check::Bool=true) where {R<:BlasReal,T<:MaybeComplex{R}} = cholesky!(copy(P); shift, check)
"""
    cholesky!(P::SPMatrix, NoPivot(); shift=0, check=true) -> Cholesky

The same as [`cholesky`](@ref), but saves space by overwriting the input `P`, instead of creating a copy.
"""
function LinearAlgebra.cholesky!(P::SPMatrix{T}, ::NoPivot=NoPivot(); shift::R=zero(R),
    check::Bool=true) where {R<:BlasReal,T<:MaybeComplex{R}}
    if !iszero(shift)
        for i in SPDiagonalIterator(P)
            @inbounds P[i] += shift
        end
    end
    C, info = pptrf!(P)
    check && LinearAlgebra.checkpositivedefinite(info)
    return Cholesky(SPMatrix(P.dim, C.data, packed_isupper(P) ? :U : :L), packed_ulchar(P), info)
end

LinearAlgebra.inv!(C::Cholesky{<:BlasFloat,<:SPMatrix}) = pptri!(C.factors)
LinearAlgebra.inv(C::Cholesky{<:BlasFloat,<:SPMatrix}) = pptri!(copy(C.factors))


"""
    bunchkaufman(P::SPMatrix, ipiv=missing; check = true) -> S::BunchKaufman

Compute the Bunch-Kaufman factorization of a packed matrix `P` `P'*U*D*U'*P` or `P'*L*D*L'*P`, depending on which triangle is
stored in `P`, and return a `BunchKaufman` object.

Iterating the decomposition produces the components `S.D`, `S.U` or `S.L` as appropriate given `S.uplo`, and `S.p`.

When `check = true`, an error is thrown if the decomposition fails. When `check = false`, responsibility for checking the
decomposition's validity (via issuccess) lies with the user.

The following functions are available for BunchKaufman objects: `size`, `\\`, `inv`, `getindex`.

The pivoting vector `ipiv` may be passed beforehand to save allocations.
"""
LinearAlgebra.bunchkaufman(P::SPMatrix{R}, ipiv::Union{AbstractVector{BlasInt},Missing}=missing;
    check::Bool=true) where {R<:BlasFloat} = bunchkaufman!(copy(P), ipiv; check)

"""
    bunchkaufman!(P::SPMatrix, ipiv=missing; check = true) -> S::BunchKaufman

The same as [`bunchkaufman`](@ref), but saves space by overwriting the input `P`, instead of creating a copy.
"""
function LinearAlgebra.bunchkaufman!(P::SPMatrix{R}, ipiv::Union{AbstractVector{BlasInt},Missing}=missing;
    check::Bool=true) where {R<:BlasFloat}
    LD, ipiv, info = R <: BlasReal ? sptrf!(P, ipiv) : hptrf!(P, ipiv)
    check && LinearAlgebra.checknonsingular(info)
    return BunchKaufman(LD, ipiv, packed_ulchar(P), R <: BlasReal, false, info)
end

LinearAlgebra.inv!(B::BunchKaufman{T,<:SPMatrix}, work::Union{<:AbstractVector{T},Missing}=missing) where {T<:BlasReal} =
    sptri!(B.LD, B.ipiv, work)
LinearAlgebra.inv!(B::BunchKaufman{T,<:SPMatrix}, work::Union{<:AbstractVector{T},Missing}=missing) where {T<:BlasComplex} =
    hptri!(B.LD, B.ipiv, work)
LinearAlgebra.inv(B::BunchKaufman{T,<:SPMatrix}, work::Union{<:AbstractVector{T},Missing}=missing) where {T<:BlasReal} =
    sptri!(copy(B.LD), B.ipiv, work)
LinearAlgebra.inv(B::BunchKaufman{T,<:SPMatrix}, work::Union{<:AbstractVector{T},Missing}=missing) where {T<:BlasComplex} =
    hptri!(copy(B.LD), B.ipiv, work)

# We need to adopt this function for BunchKaufman, as all the fields are lazily calculated
function Base.getproperty(B::BunchKaufman{T,<:SPMatrixUnscaled}, d::Symbol) where {T<:BlasFloat}
    @assert(!getfield(B, :rook))
    n = size(B, 1)
    if d === :p
        return LinearAlgebra._ipiv2perm_bk(getfield(B, :ipiv), n, getfield(B, :uplo), false)
    elseif d === :P
        return Matrix{T}(I, n, n)[:,invperm(B.p)]
    elseif d === :L || d === :U || d === :D
        LUD, od = syconv_packed!(copy(getfield(B, :LD)), getfield(B, :ipiv))
        if d === :D
            if getfield(B, :uplo) == 'L'
                odl = od[1:n - 1]
                return Tridiagonal(odl, diag(LUD), getfield(B, :symmetric) ? odl : conj.(odl))
            else # 'U'
                odu = od[2:n]
                return Tridiagonal(getfield(B, :symmetric) ? odu : conj.(odu), diag(LUD), odu)
            end
        elseif d === :L
            if getfield(B, :uplo) == 'L'
                return UnitLowerTriangular(LUD)
            else
                throw(ArgumentError("factorization is U*D*transpose(U) but you requested L"))
            end
        else # :U
            if B.uplo == 'U'
                return UnitUpperTriangular(LUD)
            else
                throw(ArgumentError("factorization is L*D*transpose(L) but you requested U"))
            end
        end
    else
        getfield(B, d)
    end
end

# LAPACK does not provide a function for packed syconv matrices, so we translate the reference implementation to Julia.
# We only implement the convert way, not the revert way.
@pmalso function syconv_packed!(uplo::AbstractChar, A::PM{T}, ipiv::AbstractVector{<:Integer},
    E::Union{AbstractVector{T},Missing}=missing) where {T}
    chkuplo(uplo)
    n = packedside(A)
    length(ipiv) != n && throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $n"))
    if ismissing(E)
        E = Vector{T}(undef, n)
    else
        length(E) != n && throw(DimensionMismatch("E has length $(length(E)), but needs $n"))
    end
    iszero(n) && return A, E
    @inbounds if uplo == 'U'
        # convert VALUE
        i = n
        E[1] = zero(T)
        while i > 1
            if ipiv[i] < 0
                E[i] = A[i-1, i]
                E[i-1] = zero(T)
                A[i-1, i] = zero(T)
                i -= 1
            else
                E[i] = zero(T)
            end
            i -= 1
        end
        # convert PERMUTATIONS
        i = n
        while i ≥ 1
            if ipiv[i] > 0
                ip = ipiv[i]
                for j in i+1:n
                    A[ip, j], A[i, j] = A[i, j], A[ip, j]
                end
            else
                ip = -ipiv[i]
                for j in i+1:n
                    A[ip, j], A[i-1, j] = A[i-1, j], A[ip, j]
                end
                i -= 1
            end
            i -= 1
        end
    else
        # convert VALUE
        i = 1
        E[n] = zero(T)
        while i ≤ n
            if i < n && ipiv[i] < 0
                E[i] = A[i+1, i]
                E[i+1] = zero(T)
                A[i+1, i] = zero(T)
                i += 1
            else
                E[i] = zero(T)
            end
            i += 1
        end
        # convert PERMUTATIONS
        i = 1
        while i ≤ n
            if ipiv[i] > 0
                ip = ipiv[i]
                for j in 1:i-1
                    A[ip, j], A[i, j] = A[i, j], A[ip, j]
                end
            else
                ip = -ipiv[i]
                for j in 1:i-1
                    A[ip, j], A[i+1, j] = A[i+1, j], A[ip, j]
                end
                i += 1
            end
            i += 1
        end
    end
    return A, E
end

# If we don't implement this, the \ operator will convert the packed matrix back to a dense matrix
BunchKaufman{T}(B::BunchKaufman{Q,<:SPMatrix} where {Q}) where {T} =
    BunchKaufman(convert(SPMatrix{T}, B.LD), B.ipiv, B.uplo, B.symmetric, B.rook, B.info)


LinearAlgebra.eigvals(P::SPMatrix{<:BlasFloat}, args...) = eigvals!(copy(P), args...)
"""
    eigvals(P::SPMatrix, args...)

Return the eigenvalues of `P`.

This function calls [`spevd!`](@ref) and will forward all additional arguments. However, consider using [`eigvals!`](@ref) for
a non-allocationg version.
"""
LinearAlgebra.eigvals(::SPMatrix)
"""
    eigvals(P::SPMatrix, vl::Real, vu::Real, args...)

Return the eigenvalues of `P`. It is possible to calculate only a subset of the eigenvalues by specifying a pair `vl` and `vu`
for the lower and upper boundaries of the eigenvalues.

This function calls [`spevx!`](@ref) and will forward all additional arguments. However, consider using [`eigvals!`](@ref) for
a non-allocationg version.
"""
LinearAlgebra.eigvals(::SPMatrix, ::Real, ::Real)
"""
    eigvals(P::SPMatrix, range::UnitRange, args...)

Return the eigenvalues of `P`, overwriting `P` in the progress. It is possible to calculate only a subset of the eigenvalues by
specifying a `UnitRange` `range` covering indices of the sorted eigenvalues, e.g. the 2nd to 8th eigenvalues.

This function calls [`spevx!`](@ref) and will forward all additional arguments. However, consider using [`eigvals!`](@ref) for
a non-allocationg version.
"""
LinearAlgebra.eigvals(::SPMatrix, ::UnitRange)
"""
    eigvals(AP::SPMatrix, BP::SPMatrix, args...)

Compute the generalized eigenvalues of `AP` and `BP`.

See also [`eigen`](@ref).
"""
LinearAlgebra.eigvals(AP::PM, BP::PM, args...) where {PM<:SPMatrix{<:BlasFloat}} = eigvals!(copy(AP), copy(BP), args...)

# for disambiguous methods, we cannot use BlasFloat here
LinearAlgebra.eigvals!(P::SPMatrix{T}, args...) where {R<:BlasReal,T<:MaybeComplex{R}} = spevd!('N', P, args...)
LinearAlgebra.eigvals!(P::SPMatrix{T}, vl::R, vu::R, args...) where {R<:BlasReal,T<:MaybeComplex{R}} =
    spevx!('N', 'V', P, vl, vu, nothing, nothing, -one(R), args...)
LinearAlgebra.eigvals!(P::SPMatrix{T}, range::UnitRange, args...) where {R<:BlasReal,T<:MaybeComplex{R}} =
    spevx!('N', 'I', P, nothing, nothing, range.start, range.stop, -one(_realtype(eltype(P))), args...)
LinearAlgebra.eigvals!(AP::SPMatrix{T}, BP::SPMatrix{T}, args...) where {T<:BlasFloat} =
    spgvd!(1, 'N', AP, BP, args...)[1]
"""
    eigvals!(P::SPMatrix, args...)
    eigvals!(P::SPMatrix, vl::Real, vu::Real, args...)
    eigvals!(P::SPMatrix, range::UnitRange, args...)

The same as [`eigvals`](@ref), but saves space by overwriting the input `P`, instead of creating a copy.
"""
LinearAlgebra.eigvals!

"""
    eigmax(P::SPMatrix, args...)

Return the largest eigenvalue of `P`.

See also [`eigmax!`](@ref), [`eigvals`](@ref).
"""
LinearAlgebra.eigmax(P::SPMatrix, args...) = eigmax!(copy(P), args...)
"""
    eigmax!(P::SPMatrix, args...)

Return the largest eigenvalue of `P`, overwriting `P` in the progress.

See also [`eigvals!`](@ref).
"""
eigmax!(P::SPMatrix, args...) = first(eigvals!(P, P.dim:P.dim, args...))
"""
    eigmin(P::SPMatrix, args...)

Return the smallest eigenvalue of `P`

See also [`eigmin!`](@ref), [`eigvals`](@ref).
"""
LinearAlgebra.eigmin(P::SPMatrix, args...) = eigmin!(copy(P), args...)
"""
    eigmin!(P::SPMatrix, args...)

Return the smallest eigenvalue of `P`, overwriting `P` in the progress.

See also [`eigvals!`](@ref).
"""
eigmin!(P::SPMatrix, args...) = first(eigvals!(P, 1:1, args...))

"""
    eigvecs(P::SPMatrix, args...)

Return a matrix `M` whose columns are the eigenvectors of `P`. (The `k`th eigenvector can be obtained from the slice
`M[:, k]`.)

See also [`eigen`](@ref).
"""
LinearAlgebra.eigvecs(P::SPMatrix, args...) = eigen(P, args...).vectors
"""
    eigvecs(A::SPMatrix, B::SPMatrix, args...)

Return a matrix `M` whose columns are the generalized eigenvectors of `A` and `B`. (The `k`th eigenvector can be obtained from
the slice `M[:, k]`.)

See also [`eigen`](@ref).
"""
LinearAlgebra.eigvecs(::SPMatrix, ::SPMatrix)
"""
    eigvecs!(P::SPMatrix, args...)
    eigvecs(A::SPMatrix, B::SPMatrix, args...)

The same as [`eigvecs`](@ref), but saves space by overwriting the input `P` (or `A` and `B`), instead of creating a copy.
"""
eigvecs!(P::SPMatrix, args...) = eigen!(P, args...).vectors

"""
    eigen(P::SPMatrix, args...) -> Eigen

Compute the eigenvalue decomposition of `P`, returning an `Eigen` factorization object `F` which contains the eigenvalues in
`F.values` and the eigenvectors in the columns of the matrix `F.vectors`. (The `k`th eigenvector can be obtained from the slice
`F.vectors[:, k]`.)

This function calls [`spevd!`](@ref) and will forward all arguments. However, consider using [`eigen!`](@ref) for a
non-allocating version.
"""
LinearAlgebra.eigen(P::SPMatrix{<:BlasFloat}, args...) = eigen!(copy(P), args...)
"""
    eigen(A::SPMatrix, B::SPMatrix, args...) -> GeneralizedEigen

Compute the generalized eigenvalue decomposition of `A` and `B`, returning a `GeneralizedEigen` factorization object `F` which
contains the generalized eigenvalues in `F.values` and the generalized eigenvectors in the columns of the matrix `F.vectors`.
(The `k`th generalized eigenvector can be obtained from the slice `F.vectors[:, k]`.)

This function calls [`spgvd!`](@ref) and will forward all arguments. However, consider using [`eigen!`](@ref) for a
non-allocating version.
"""
LinearAlgebra.eigen(A::SPMatrix{T}, B::SPMatrix{T}, args...) where {T<:BlasFloat} = eigen!(copy(A), copy(B), args...)
"""
    eigen(P::SPMatrix, irange::UnitRange, args...)

Compute the eigenvalue decomposition of `P`, returning an `Eigen` factorization object `F` which contains the eigenvalues in
`F.values` and the eigenvectors in the columns of the matrix `F.vectors`. (The `k`th eigenvector can be obtained from the slice
`F.vectors[:, k]`.)

The `UnitRange` `irange` specifies indices of the sorted eigenvalues to search for.

This function calls [`spevx!`](@ref) and will forward all arguments. However, consider using [`eigen!`](@ref) for a
non-allocating version.
"""
LinearAlgebra.eigen(::SPMatrix, ::UnitRange)
"""
    eigen(P::SPMatrix, vl::Real, vu::Real, args...)

Compute the eigenvalue decomposition of `P`, returning an `Eigen` factorization object `F` which contains the eigenvalues in
`F.values` and the eigenvectors in the columns of the matrix `F.vectors`. (The `k`th eigenvector can be obtained from the slice
`F.vectors[:, k]`.)

`vl` is the lower bound of the window of eigenvalues to search for, and `vu` is the upper bound.

This function calls [`spevx!`](@ref) and will forward all arguments. However, consider using [`eigen!`](@ref) for a
    non-allocating version.
"""
LinearAlgebra.eigen(::SPMatrix, ::Real, ::Real)

"""
    eigen!(P::SPMatrix, args...)
    eigen!(A::SPMatrix, B::SPMatrix, args...)

Same as [`eigen`](@ref), but saves space by overwriting the input `A` (and `B`) instead of creating a copy.
"""
LinearAlgebra.eigen!(P::SPMatrix{T}, args...) where {R<:BlasReal,T<:MaybeComplex{R}} = Eigen(spevd!('V', P, args...)...)
function LinearAlgebra.eigen!(P::SPMatrix{T}, irange::UnitRange, args...) where {R<:BlasReal,T<:MaybeComplex{R}}
    W, Z, info, _ = spevx!('V', 'I', P, nothing, nothing, irange.start, irange.stop, -one(_realtype(eltype(P))), args...)
    chklapackerror(info)
    return Eigen(W, Z)
end
function LinearAlgebra.eigen!(P::SPMatrix{T}, vl::R, vu::R, args...) where {R<:BlasReal,T<:MaybeComplex{R}}
    W, Z, info, _ = spevx!('V', 'V', P, vl, vu, nothing, nothing, -one(R), args...)
    chklapackerror(info)
    return Eigen(W, Z)
end
LinearAlgebra.eigen!(AP::SPMatrix{T}, BP::SPMatrix{T}, args...) where {T<:BlasFloat} =
    GeneralizedEigen(spgvd!(1, 'V', AP, BP, args...)[1:2]...)


"""
    diagind(P::SPMatrix, k::Integer=0)

A vector containing the indices of the `k`the diagonal of the packed matrix `P`.

See also: [`diag`](@ref), [`SPDiagonalIterator`](@ref).
"""
LinearAlgebra.diagind(P::SPMatrix, k::Integer=0) = collect(SPDiagonalIterator(P, k))

"""
    diag(P::SPMatrix, k::Integer=0)

The `k`th diagonal of a packed matrix, as a vector.

See also: [`diagind`](@ref), [`SPDiagonalIterator`](@ref).
"""
function LinearAlgebra.diag(P::SPMatrix{R}, k::Integer=0) where {R}
    iter = SPDiagonalIterator(P, k)
    diagonal = Vector{R}(undef, length(iter))
    for (i, idx) in enumerate(iter)
        @inbounds diagonal[i] = P[idx]
    end
    return diagonal
end


function normapply(f, P::SPMatrix{R}, init::T=zero(_realtype(R))) where {R,T}
    result::T = init
    diags = SPDiagonalIterator(P)
    cur_diag = iterate(diags)
    i = 1
    @inbounds while !isnothing(cur_diag)
        @simd for j in i:cur_diag[1]-1
            if packed_isscaled(P)
                result = f(result, P.data[j] * sqrt(inv(R(2))), false)
            else
                result = f(result, P.data[j], false)
            end
        end
        result = f(result, P.data[cur_diag[1]], true)
        i = cur_diag[1] +1
        cur_diag = iterate(diags, cur_diag[2])
    end
    return result
end
function normapply(f, P::SPMatrix{R,<:SparseVector}) where {R}
    nzs = rowvals(P.data)
    vs = nonzeros(P.data)
    diags = SPDiagonalIterator(P)
    cur_diag = iterate(diags)
    isnothing(cur_diag) && return zero(R)
    result = zero(_realtype(R))
    @inbounds for i in 1:length(nzs)
        j = nzs[i]
        while cur_diag[1] < j
            cur_diag = iterate(diags, cur_diag[2])
            if isnothing(cur_diag)
                cur_diag = (typemax(Int), cur_diag[2])
            end
        end
        if cur_diag[1] == j
            result = f(result, vs[i], true)
        elseif packed_isscaled(P)
            result = f(result, vs[i] * sqrt(inv(R(2))), false)
        else
            result = f(result, vs[i], false)
        end
    end
    return result
end
LinearAlgebra.norm2(P::SPMatrixUnscaled) = sqrt(normapply((Σ, x, diag) -> Σ + (diag ? norm(x)^2 : 2norm(x)^2), P))
LinearAlgebra.norm2(P::SPMatrixScaled) = LinearAlgebra.norm2(P.data)
LinearAlgebra.norm2(P::SPMatrixScaled{R,<:SparseVector} where {R}) = LinearAlgebra.norm2(nonzeros(P.data))
LinearAlgebra.norm1(P::SPMatrix) = normapply((Σ, x, diag) -> Σ + (diag ? norm(x) : 2norm(x)), P)
LinearAlgebra.normInf(P::SPMatrixUnscaled) = LinearAlgebra.normInf(P.data)
LinearAlgebra.normInf(P::SPMatrixScaled) = normapply((m, x, diag) -> max(m, norm(x)), P)
Base._simple_count(f, P::SPMatrix, init::T) where {T} =
    normapply((Σ, x, diag) -> f(x) ? (diag ? Σ + one(T) : Σ + one(T) + one(T)) : Σ, P, init)
LinearAlgebra.normMinusInf(P::SPMatrixUnscaled) = LinearAlgebra.normMinusInf(P.data)
LinearAlgebra.normMinusInf(P::SPMatrixScaled{R}) where {R} = normapply((m, x, diag) -> min(m, norm(x)), P,
    _realtype(R)(Inf))
LinearAlgebra.normp(P::SPMatrix, p) = normapply((Σ, x, diag) -> Σ + (diag ? norm(x)^p : 2norm(x)^p), P)^(1/p)

"""
    norm(P::SPMatrix, p::Real=2)

For any packed matrix `P`, compute the `p`-norm as if `P` was interpreted as a full vector (containing the off-diagonals twice
without scaling).

The `p`-norm is defined as
```math
    \\lVert P\\rVert_p = \\Biggl( \\sum_{i, j = 1}^n \\lvert P_{i, j}\\rvert^p \\Biggr)^{1/p}
```
with ``P_{i, j}`` the entries of ``P``, ``\\lvert P_{i, j}\\rvert`` the `norm` of ``P_{i, j}``, and ``n`` its side dimension.

`p` can assume any numeric value (even though not all values produce a mathematically valid vector norm). In particular,
`norm(P, Inf)` returns the largest value in `abs.(P)`, whereas `norm(P, -Inf)` returns the smallest. If `p=2`, then this is
equivalent to the Frobenius norm.

The second argument `p` is not necessarily a part of the interface for `norm`, i.e. a custom type may only implement `norm(P)`
without second argument.
"""
LinearAlgebra.norm(::SPMatrix, ::Real=2)

"""
    tr(P::SPMatrix)

Matrix trace. Sums the diagonal elements of `P`.
"""
function LinearAlgebra.tr(P::SPMatrix{R}) where {R}
    trace = zero(R)
    for idx in SPDiagonalIterator(P)
        @inbounds trace += P[idx]
    end
    return trace
end


LinearAlgebra.issymmetric(::SPMatrix{<:Real}) = true

"""
    isposdef(P::SPMatrix, tol=0)

Test whether a matrix is positive definite by trying to perform a Cholesky factorization of `P`.

See also [`isposdef!`](@ref), [`cholesky`](@ref).
"""
LinearAlgebra.isposdef(P::SPMatrix{T}, tol::R=zero(R)) where {R<:BlasReal,T<:MaybeComplex{R}} = isposdef!(copy(P), tol)
"""
    isposdef!(P::SPMatrix, tol=0)

Test whether a matrix is positive definite by trying to perform a Cholesky factorization of `P`, overwriting `P` in the
process. See also [`isposdef`](@ref).
"""
LinearAlgebra.isposdef!(P::SPMatrix{T}, tol::R=zero(R)) where {R<:BlasReal,T<:MaybeComplex{R}} =
    isposdef(cholesky!(P, shift=tol, check=false))

LinearAlgebra.ishermitian(::SPMatrix) = true


"""
    transpose(P::SPMatrix{<:Real})

Identity operation for real-valued packed matrices.
"""
Base.transpose(P::SPMatrix{<:Real}) = P
"""
    P'
    adjoint(P::SPMatrix)

Identity operation for packed matrices
"""
LinearAlgebra.adjoint(P::SPMatrix) = P

"""
    checksquare(P::SPMatrix)

Returns the side dimension of `P`.
"""
LinearAlgebra.checksquare(P::SPMatrix) = P.dim


# These are for multiplications where the matrix is directly interpreted as a vector.
"""
    mul!(::SPMatrix, B::AbstractMatrix, ::AbstractVector, α, β)
    mul!(::SPMatrix, B::AbstractMatrix, ::SPMatrix, α, β)
    mul!(::AbstractVector, B::AbstractMatrix, ::SPMatrix, α, β)

Combined inplace matrix-vector multiply-add ``A B α + C β``. The result is stored in `C` by overwriting it. Note that `C` must
not be aliased with either `A` or `B`.
This method will automatically interpret all `SPMatrix` arguments as their (scaled) vectorizations (so this is simply an
alias to calling the standard `mul!` method on the [`vec`](@ref) of the packed matrices).

Note that `B` must not be a packed matrix.
"""
LinearAlgebra.mul!(C::SPMatrix, A::AbstractMatrix, B::AbstractVector, α::Number, β::Number) = mul!(C.data, A, B, α, β)
LinearAlgebra.mul!(C::SPMatrix, A::AbstractMatrix, B::SPMatrix, α::Number, β::Number) = mul!(C.data, A, B.data, α, β)
LinearAlgebra.mul!(C::AbstractVector, A::AbstractMatrix, B::SPMatrix, α::Number, β::Number) = mul!(C, A, B.data, α, β)
# But we need to be very careful: if A above was an SPMatrix, they all transform to Vector, SPMatrix, Vector, which will
# (potentially) call the mul! below that performs symmetric multiplication - which is not what should be done.
LinearAlgebra.mul!(::Union{<:AbstractVector,SPMatrix}, ::SPMatrix, ::Union{<:AbstractVector,<:SPMatrix}, ::Number,
    ::Number) = throw(DimensionMismatch())
# And this is the symmetric multiplication. BLAS only offers it for the packed format. So either we write a helper that
# - allocates a copy and unscales it (bad - mul! shouldn't allocate) or
# - unscales the matrix, multiplies and rescales (bad - floating point not necessarily reversible; not threadsafe)
# or we leave this unsupported.
"""
    mul!(C::AbstractVector, A::SPMatrixUnscaled, B::AbstractVector, α, β)

Combined inplace matrix-vector multiply-add `A B α + C β`. The result is stored in `C` by overwriting it. Note that `C` must
not be aliased with either `A` or `B`.
This method requires an unscaled packed matrix and only works on native floating point types supported by BLAS.

See also [`spmv!`](@ref), [`hpmv!`](@ref).
"""
LinearAlgebra.mul!(C::AbstractVector, A::SPMatrixUnscaled{R}, B::AbstractVector, α::Number, β::Number) where {R<:BlasReal} =
    spmv!(α, A, B, β, C)
LinearAlgebra.mul!(C::AbstractVector, A::SPMatrixUnscaled{R}, B::AbstractVector, α::Number, β::Number) where {R<:BlasComplex} =
    hpmv!(α, A, B, β, C)

"""
    lmul!(a::Number, P::SPMatrix)

Scale a packed matrix `P` by a scalar `a` overwriting `P` in-place. Use [`rmul!`](@ref) to multiply scalar from right. The
scaling operation respects the semantics of the multiplication `*` between `a` and an element of `P`. In particular, this also
applies to multiplication involving non-finite numbers such as `NaN` and `±Inf`.
"""
LinearAlgebra.lmul!(a::Number, P::SPMatrix) = (lmul!(a, P.data); P)

"""
    rmul!(P::SPMatrix, b::Number)

Scale a packed matrix `P` by a scalar `b` overwriting `P` in-place. Use [`lmul!`](@ref) to multiply scalar from left. The
scaling operation respects the semantics of the multiplication `*` between an element of `P` and `b`. In particular, this also
applies to multiplication involving non-finite numbers such as `NaN` and `±Inf`.
"""
LinearAlgebra.rmul!(P::SPMatrix, b::Number) = (rmul!(P.data, b); P)

"""
    ldiv!(P, B)

Compute `P \\ B` in-place and overwriting `B` to store the result.

The argument `P` should not be a matrix. Rather, instead of matrices it should be a factorization object (e.g. produced by
[`cholesky`](@ref) or [`bunchkaufman`](@ref) from a `SPMatrix`). The reason for this is that factorization itself is both
expensive and typically allocates memory (although it can also be done in-place via, e.g., [`cholesky!`](@ref)), and
performance-critical situations requiring `ldiv!` usually also require fine-grained control over the factorization of `P`.
"""
LinearAlgebra.ldiv!(C::Cholesky{T,<:SPMatrix}, B::StridedVecOrMat{T}) where {T<:BlasFloat} = pptrs!(C.factors, B)
LinearAlgebra.ldiv!(B::BunchKaufman{T,<:SPMatrix}, R::StridedVecOrMat{T}) where {T<:BlasReal} = sptrs!(B.LD, B.ipiv, R)
LinearAlgebra.ldiv!(B::BunchKaufman{T,<:SPMatrix}, R::StridedVecOrMat{T}) where {T<:BlasComplex} = hptrs!(B.LD, B.ipiv, R)