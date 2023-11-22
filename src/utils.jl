export packedsize, PackedDiagonalIterator, rmul_diags!, rmul_offdiags!, lmul_diags!, lmul_offdiags!

@doc raw"""
    packedsize(dim)

Calculates the number of unique entries in a symmetric matrix, ``\frac{\mathit{dim}(\mathit{dim} +1)}{2}``.
"""
@inline packedsize(dim) = dim * (dim +1) ÷ 2
chkpacked(n::Integer, AP::AbstractVector) = 2length(AP) == n * (n +1) ||
    error("Packed storage length does not match dimension")

@inline rowcol_to_vec(P::PackedMatrixUpper, row, col) =
    (@boundscheck(1 ≤ row ≤ col || throw(BoundsError(P, (row, col)))); return col * (col -1) ÷ 2 + row)
@inline rowcol_to_vec(P::PackedMatrixLower, row, col) =
    (@boundscheck(1 ≤ col ≤ row || throw(BoundsError(P, (row, col)))); return (2P.dim - col) * (col -1) ÷ 2 + row)

"""
    PackedDiagonalIterator(P::PackedMatrix, k=0)
    PackedDiagonalIterator(fmt::Symbol, dim, k=0)

Creates an iterator that returns the linear indices for iterating through the `k`th diagonal of a packed matrix.
"""
struct PackedDiagonalIterator{Fmt}
    dim::Int
    k::UInt
    PackedDiagonalIterator(P::PackedMatrixUpper, k=0) = new{:U}(P.dim, abs(k))
    PackedDiagonalIterator(P::PackedMatrixLower, k=0) = new{:L}(P.dim, abs(k))
    function PackedDiagonalIterator(fmt::Symbol, dim, k=0)
        fmt === :U || fmt === :L || error("Invalid symbol for diagonal iterator construction")
        new{fmt}(dim, abs(k))
    end
end
function Base.iterate(iter::PackedDiagonalIterator{:U})
    if iter.k ≥ iter.dim
        return nothing
    else
        j = iter.k * (iter.k +1) ÷ 2 +1
        return j, (j, iter.k +2)
    end
end
function Base.iterate(iter::PackedDiagonalIterator{:L})
    if iter.k ≥ iter.dim
        return nothing
    else
        j = iter.k +1
        return j, (j, iter.dim)
    end
end
function Base.iterate(iter::PackedDiagonalIterator{:U}, state)
    j, δ = state
    j += δ
    if δ > iter.dim
        return nothing
    else
        return j, (j, δ +1)
    end
end
function Base.iterate(::PackedDiagonalIterator{:L}, state)
    j, δ = state
    if isone(δ)
        return nothing
    else
        j += δ
        return j, (j, δ -1)
    end
end
Base.IteratorSize(::Type{PackedDiagonalIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{PackedDiagonalIterator}) = Base.HasEltype()
Base.eltype(::PackedDiagonalIterator) = Int
Base.length(iter::PackedDiagonalIterator) = iter.dim - iter.k


"""
    rmul_diags!(P::PackedMatrix{R}, factor::R) where {R}

Right-multiplies all diagonal entries in `P` by `factor`.

See also [`lmul_diags!`](@ref), [`rmul_offdiags!`](@ref), [`lmul_offdiags!`](@ref).
"""
function rmul_diags!(P::PackedMatrix{R}, factor::R) where {R}
    data = P.data
    for i in PackedDiagonalIterator(P)
        @inbounds data[i] = data[i] * factor
    end
    return P
end
"""
    rmul_offdiags!(P::PackedMatrix{R}, factor::R) where {R}

Right-multiplies all off-diagonal entries in `P` by `factor`.

See also [`rmul_diags!`](@ref), [`lmul_diags!`](@ref), [`lmul_offdiags!`](@ref).
"""
function rmul_offdiags!(P::PackedMatrix{R}, factor::R) where {R}
    diags = PackedDiagonalIterator(P)
    data = P.data
    for (d₁, d₂) in zip(diags, Iterators.drop(diags, 1))
        @inbounds rmul!(@view(data[d₁+1:d₂-1]), factor)
    end
    return P
end
"""
    lmul_diags!(P::PackedMatrix{R}, factor::R) where {R}

Left-multiplies all diagonal entries in `P` by `factor`.

See also [`rmul_diags!`](@ref), [`rmul_offdiags!`](@ref), [`lmul_offdiags!`](@ref).
"""
function lmul_diags!(P::PackedMatrix{R}, factor::R) where {R}
    data = P.data
    for i in PackedDiagonalIterator(P)
        @inbounds data[i] = factor * data[i]
    end
    return P
end
"""
    lmul_offdiags!(P::PackedMatrix{R}, factor::R) where {R}

Left-multiplies all diagonal entries in `P` by `factor`.

See also [`rmul_diags!`](@ref), [`rmul_offdiags!`](@ref), [`lmul_diags!`](@ref).
"""
function lmul_offdiags!(P::PackedMatrix{R}, factor::R) where {R}
    diags = PackedDiagonalIterator(P)
    data = P.data
    for (d₁, d₂) in zip(diags, Iterators.drop(diags, 1))
        @inbounds lmul!(@view(data[d₁+1:d₂-1]), factor)
    end
    return P
end