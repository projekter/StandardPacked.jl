export eigmin!, eigmax!

LinearAlgebra.checksquare(P::PackedMatrix) = P.dim

LinearAlgebra.diagind(P::PackedMatrix, k::Integer=0) = collect(PackedDiagonalIterator(P, k))

function LinearAlgebra.diag(P::PackedMatrix{R}, k::Integer=0) where {R}
    iter = PackedDiagonalIterator(P, k)
    diagonal = Vector{R}(undef, length(iter))
    for (i, idx) in enumerate(iter)
        @inbounds diagonal[i] = P[idx]
    end
    return diagonal
end
function LinearAlgebra.tr(P::PackedMatrix{R}) where {R}
    trace = zero(R)
    for idx in PackedDiagonalIterator(P)
        @inbounds trace += P[idx]
    end
    return trace
end

"""
    vec(P::PackedMatrix)

Returns the vectorized data associated with `P`. Note that this returns the actual vector, not a copy.
"""
LinearAlgebra.vec(P::PackedMatrix) = P.data

"""
    axpy!(::Number, Union{<:PackedMatrix,<:AbstractVector}, Union{<:PackedMatrix,<:AbstractVector})

Convenience wrapper that automatically translates the involved packed matrices to their corresponding vectorizations.
"""
LinearAlgebra.axpy!(a::Number, x::PackedMatrix, y::AbstractVector) = LinearAlgebra.axpy!(a, x.data, y)
LinearAlgebra.axpy!(a::Number, x::AbstractVector, y::PackedMatrix) = LinearAlgebra.axpy!(a, x, y.data)
LinearAlgebra.axpy!(a::Number, x::PackedMatrix, y::PackedMatrix) = LinearAlgebra.axpy!(a, x.data, y.data)

# These are for multiplications where the matrix is directly interpreted as a vector.
"""
    mul!(::PackedMatrix, B::AbstractMatrix, ::AbstractVector, α, β)
    mul!(::PackedMatrix, B::AbstractMatrix, ::PackedMatrix, α, β)
    mul!(::AbstractVector, B::AbstractMatrix, ::PackedMatrix, α, β)

Convenience wrappers that automatically translate the involved packed matrices to their corresponding vectorizations.
Note that `B` is not allowed to be a `PackedMatrix`.
"""
LinearAlgebra.mul!(C::PackedMatrix, A::AbstractMatrix, B::AbstractVector, α::Number, β::Number) = mul!(C.data, A, B, α, β)
LinearAlgebra.mul!(C::PackedMatrix, A::AbstractMatrix, B::PackedMatrix, α::Number, β::Number) = mul!(C.data, A, B.data, α, β)
LinearAlgebra.mul!(C::AbstractVector, A::AbstractMatrix, B::PackedMatrix, α::Number, β::Number) = mul!(C, A, B.data, α, β)
# But we need to be very careful: if A above was a PackedMatrix, they all transform to Vector, PackedMatrix, Vector, which will
# (potentially) call the mul! below that performs symmetric multiplication - which is not what should be done.
LinearAlgebra.mul!(::Union{<:AbstractVector,PackedMatrix}, ::PackedMatrix, ::Union{<:AbstractVector,<:PackedMatrix}, ::Number,
    ::Number) = throw(DimensionMismatch())
# And this is the symmetric multiplication. BLAS only offers it for the packed format. So either we write a helper that
# - allocates a copy and unscales it (bad - mul! shouldn't allocate) or
# - unscales the matrix, multiplies and rescales (bad - floating point not necessarily reversible; not threadsafe)
# or we leave this unsupported.
"""
    mul!(C::AbstractVector, A::PackedMatrixUnscaled, B::AbstractVector, α, β)

Combined inplace matrix-vector multiply-add `A B α + C β`. The result is stored in `C` by overwriting it. Note that `C` must
not be aliased with either `A` or `B`.
This function requires an unscaled packed matrix and only works on native floating point types supported by BLAS.

See also [`spmv!`](@ref).
"""
LinearAlgebra.mul!(C::AbstractVector, A::PackedMatrixUnscaled, B::AbstractVector, α::Number, β::Number) =
    spmv!(packed_ulchar(A), α, A.data, B, β, C)
# Also a wrapper for rank-one updates. Here, we are allowed to mutate AP, so we un-scale it. But beware, the type may change!
@doc raw"""
    spr(α, x::AbstractVector, P::PackedMatrix)

Convenience wrapper for the symmetric rank 1 operation ``P := \alpha x x^\top + P``.
This function will convert `P` to an unscaled matrix if necessary and only works on native floating point types supported by
BLAS. It returns the unscaled result.

See also [`spr!`](@ref).
"""
function spr!(α, x::AbstractVector, P::PackedMatrix)
    Pu = packed_unscale!(P)
    spr!(packed_ulchar(Pu), α, x, Pu.data)
    return Pu
end

"""
    dot(::PackedMatrix, ::PackedMatrix)

Gives the scalar product between two packed matrices, which corresponds to the Frobenius/Hilbert-Schmidt scalar product for
matrices. This is fastest for scaled matrices.
"""
function LinearAlgebra.dot(A::PackedMatrix{R1,V,Fmt} where {V}, B::PackedMatrix{R2,V,Fmt} where {V}) where {R1,R2,Fmt}
    A.dim == B.dim || error("Matrices must have same dimensions")
    if packed_isscaled(Fmt)
        return dot(A.data, B.data)
    else
        result = zero(promote_type(R1, R2))
        diags = PackedDiagonalIterator(A)
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
function LinearAlgebra.dot(A::PackedMatrix{R1,<:SparseVector,Fmt}, B::PackedMatrix{R2,V,Fmt} where {V}) where {R1,R2,Fmt}
    A.dim == B.dim || error("Matrices must have same dimensions")
    if packed_isscaled(Fmt)
        return dot(A.data, B.data)
    else
        result = zero(promote_type(R1, R2))
        nzs = rowvals(A.data)
        vs = nonzeros(A.data)
        diags = PackedDiagonalIterator(A)
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
LinearAlgebra.dot(A::PackedMatrix{R}, B::PackedMatrix{R,<:SparseVector}) where {R} = conj(dot(B, A))
function LinearAlgebra.dot(A::PackedMatrix{R1,<:SparseVector,Fmt}, B::PackedMatrix{R2,<:SparseVector,Fmt}) where {R1,R2,Fmt}
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
        diags = PackedDiagonalIterator(A)
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

function normapply(f, P::PackedMatrix{R}, init::T=zero(R)) where {R,T}
    result::T = init
    diags = PackedDiagonalIterator(P)
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
function normapply(f, P::PackedMatrix{R,<:SparseVector}) where {R}
    nzs = rowvals(P.data)
    vs = nonzeros(P.data)
    diags = PackedDiagonalIterator(P)
    cur_diag = iterate(diags)
    isnothing(cur_diag) && return zero(R)
    result = zero(R)
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
LinearAlgebra.norm2(P::PackedMatrixUnscaled) = sqrt(normapply((Σ, x, diag) -> Σ + (diag ? abs(x)^2 : 2abs(x)^2), P))
LinearAlgebra.norm2(P::PackedMatrixScaled) = LinearAlgebra.norm2(P.data)
LinearAlgebra.norm2(P::PackedMatrixScaled{R,<:SparseVector}) where {R} = LinearAlgebra.norm2(nonzeros(P.data))
LinearAlgebra.norm1(P::PackedMatrix) = normapply((Σ, x, diag) -> Σ + (diag ? abs(x) : 2abs(x)), P)
LinearAlgebra.normInf(P::PackedMatrixUnscaled) = LinearAlgebra.normInf(P.data)
LinearAlgebra.normInf(P::PackedMatrixScaled) = normapply((m, x, diag) -> max(m, abs(x)), P)
Base._simple_count(f, P::PackedMatrix, init::T) where {T} =
    normapply((Σ, x, diag) -> f(x) ? (diag ? Σ + one(T) : Σ + one(T) + one(T)) : Σ, P, init)
LinearAlgebra.normMinusInf(P::PackedMatrixUnscaled) = LinearAlgebra.normMinusInf(P.data)
LinearAlgebra.normMinusInf(P::PackedMatrixScaled{R}) where {R} = normapply((m, x, diag) -> min(m, abs(x)), P, R(Inf))
LinearAlgebra.normp(P::PackedMatrix, p) = normapply((Σ, x, diag) -> Σ + (diag ? abs(x)^p : 2abs(x)^p), P)^(1/p)

"""
    eigen!(P::PackedMatrix, args...)

Compute the eigenvalue decomposition of `P`, returning an `Eigen` factorization object `F` which contains the eigenvalues in
`F.values` and the eigenvectors in the columns of the matrix `F.vectors`. (The `k`th eigenvector can be obtained from the slice
`F.vectors[:, k]`.)

This function calls [`spevd!`](@ref) and will forward all keyword arguments, which allow this operation to be truely
non-allocating.
"""
LinearAlgebra.eigen!(P::PackedMatrixUnscaled{R}, args...) where {R<:Real} =
    Eigen(spevd!('V', packed_ulchar(P), P.dim, P.data)..., args...)
function LinearAlgebra.eigen!(P::PackedMatrixScaled{R}, args...) where {R<:Real}
    fac = sqrt(R(2))
    eval, evec = spevd!('V', packed_ulchar(P), P.dim, rmul_diags!(P, fac).data, args...)
    return Eigen(rmul!(eval, inv(fac)), evec)
end
"""
    eigvals(P::PackedMatrix, args...)

Return the eigenvalues of `P`.

This function calls [`spevd!`](@ref) and will forward all keyword arguments. However, consider using [`eigvals!`](@ref) for a
non-allocating version.
"""
LinearAlgebra.eigvals(P::PackedMatrixUnscaled{R}, args...) where {R<:Real} =
    spevd!('N', packed_ulchar(P), P.dim, copy(P.data), args...)
function LinearAlgebra.eigvals(P::PackedMatrixScaled{R}, args...) where {R<:Real}
    fac = sqrt(R(2))
    return rmul!(spevd!('N', packed_ulchar(P), P.dim, rmul_diags!(copy(P), fac).data, args...), inv(fac))
end
"""
    eigen!(P::PackedMatrix{R}, vl::R, vu::R, args...) where {R}

Compute the eigenvalue decomposition of `P`, returning an `Eigen` factorization object `F` which contains the eigenvalues in
`F.values` and the eigenvectors in the columns of the matrix `F.vectors`. (The `k`th eigenvector can be obtained from the slice
`F.vectors[:, k]`.)

`vl` is the lower bound of the window of eigenvalues to search for, and `vu` is the upper bound.

This function calls [`spevx!`](@ref) and will forward all keyword arguments, which allow this operation to be truely
non-allocating.
"""
LinearAlgebra.eigen!(P::PackedMatrixUnscaled{R}, vl::R, vu::R, args...) where {R<:Real} =
    Eigen(spevx!('V', 'V', packed_ulchar(P), P.dim, P.data, vl, vu, 0, 0, -one(R), args...)...)
function LinearAlgebra.eigen!(P::PackedMatrixScaled{R}, vl::R, vu::R, args...) where {R<:Real}
    fac = sqrt(R(2))
    eval, evec = spevx!('V', 'V', packed_ulchar(P), P.dim, rmul_diags!(P, fac).data, vl * fac, vu * fac, 0, 0, -one(R),
        args...)
    return Eigen(rmul!(eval, inv(fac)), evec)
end
"""
    eigen!(P::PackedMatrix{R}, range::UnitRange, args...) where {R}

Compute the eigenvalue decomposition of `P`, returning an `Eigen` factorization object `F` which contains the eigenvalues in
`F.values` and the eigenvectors in the columns of the matrix `F.vectors`. (The `k`th eigenvector can be obtained from the slice
`F.vectors[:, k]`.)

The `UnitRange` `range` specifies indices of the sorted eigenvalues to search for.

This function calls [`spevx!`](@ref) and will forward all keyword arguments, which allow this operation to be truely
non-allocating.
"""
LinearAlgebra.eigen!(P::PackedMatrixUnscaled{R}, range::UnitRange, args...) where {R<:Real} =
    Eigen(spevx!('V', 'I', packed_ulchar(P), P.dim, P.data, zero(R), zero(R), range.start, range.stop, -one(R), args...)...)
function LinearAlgebra.eigen!(P::PackedMatrixScaled{R}, range::UnitRange, args...) where {R<:Real}
    fac = sqrt(R(2))
    eval, evec = spevx!('V', 'I', packed_ulchar(P), P.dim, rmul_diags!(P, fac).data, zero(R), zero(R), range.start,
        range.stop, -one(R), args...)
    return Eigen(rmul!(eval, inv(fac)), evec)
end
"""
    eigvals!(P::PackedMatrix, args...)

Return the eigenvalues of `P`, overwriting `P` in the progress.

This function calls [`spevd!`](@ref) and will forward all keyword arguments, which allow this operation to be truely
non-allocating.
"""
LinearAlgebra.eigvals!(P::PackedMatrixUnscaled{R}, args...) where {R<:Real} =
    spevd!('N', packed_ulchar(P), P.dim, P.data, args...)
"""
    eigvals!(P::PackedMatrix, vl::Real, vu::Real, args...)

Return the eigenvalues of `P`. It is possible to calculate only a subset of the eigenvalues by specifying a pair `vl` and `vu`
for the lower and upper boundaries of the eigenvalues.

This function calls [`spevx!`](@ref) and will forward all keyword arguments, which allow this operation to be truely
non-allocating.
"""
LinearAlgebra.eigvals!(P::PackedMatrixUnscaled{R}, vl::R, vu::R, args...) where {R<:Real} =
    spevx!('N', 'V', packed_ulchar(P), P.dim, P.data, vl, vu, 0, 0, -one(R), args...)[1]
function LinearAlgebra.eigvals!(P::PackedMatrixScaled{R}, vl::R, vu::R, args...) where {R<:Real}
    fac = sqrt(R(2))
    return rmul!(spevx!('N', 'V', packed_ulchar(P), P.dim, rmul_diags!(P, fac).data, vl * fac, vu * fac, 0, 0, -one(R),
        args...)[1], inv(fac))
end
"""
    eigvals!(P::PackedMatrix, range::UnitRange, args...)

Return the eigenvalues of `P`. It is possible to calculate only a subset of the eigenvalues by specifying a `UnitRange` `range`
covering indices of the sorted eigenvalues, e.g. the 2nd to 8th eigenvalues.

This function calls [`spevx!`](@ref) and will forward all keyword arguments, which allow this operation to be truely
non-allocating.
"""
LinearAlgebra.eigvals!(P::PackedMatrixUnscaled{R}, range::UnitRange, args...) where {R<:Real} =
    spevx!('N', 'I', packed_ulchar(P), P.dim, P.data, zero(R), zero(R), range.start, range.stop, -one(R), args...)[1]
function LinearAlgebra.eigvals!(P::PackedMatrixScaled{R}, range::UnitRange, args...) where {R<:Real}
    fac = sqrt(R(2))
    return rmul!(spevx!('N', 'I', packed_ulchar(P), P.dim, rmul_diags!(P, fac).data, zero(R), zero(R), range.start,
        range.stop, -one(R), args...)[1], inv(fac))
end
"""
    eigmin!(P::PackedMatrix, args...)

Return the smallest eigenvalue of `P`, overwriting `P` in the progress.

See also [`eigvals!`](@ref).
"""
eigmin!(P::PackedMatrix, args...) = eigvals!(P, 1:1, args...)[1]
"""
    eigmax!(P::PackedMatrix, args...)

Return the largest eigenvalue of `P`, overwriting `P` in the progress.

See also [`eigvals!`](@ref).
"""
eigmax!(P::PackedMatrix, args...) = eigvals!(P, P.dim:P.dim, args...)[1]
LinearAlgebra.eigmin(P::PackedMatrix) = eigmin!(copy(P))
LinearAlgebra.eigmax(P::PackedMatrix) = eigmax!(copy(P))
"""
    cholesky(P::PackedMatrix{R}, NoPivot(); shift=zero(R), check=true) -> Cholesky

Compute the Cholesky factorization of a packed positive definite matrix `P` and return a `Cholesky` factorization.

The triangular Cholesky factor can be obtained from the factorization `F` via `F.L` and `F.U`, where
`P ≈ F.U' * F.U ≈ F.L * F.L'`.
"""
function LinearAlgebra.cholesky!(P::PackedMatrix{R}, ::NoPivot = NoPivot(); shift::R = zero(R),
    check::Bool = true) where {R<:Real}
    if !iszero(shift)
        for i in PackedDiagonalIterator(P)
            @inbounds P[i] += shift
        end
    end
    C, info = pptrf!(packed_ulchar(P), P.dim, packed_unscale!(P).data)
    check && LinearAlgebra.checkpositivedefinite(info)
    return Cholesky(PackedMatrix(P.dim, C, packed_isupper(P) ? :U : :L), packed_ulchar(P), info)
end
"""
    isposdef(P::PackedMatrix)

Test whether a matrix is positive definite by trying to perform a Cholesky factorization of `P`.
"""
LinearAlgebra.isposdef(P::PackedMatrix{R}, tol::R=zero(R)) where {R<:Real} =
    isposdef(cholesky!(copy(P), shift=tol, check=false))