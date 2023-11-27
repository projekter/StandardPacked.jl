export PackedMatrix, PackedMatrixUpper, PackedMatrixLower, PackedMatrixUnscaled, PackedMatrixScaled, packed_isupper,
    packed_islower, packed_isscaled, packed_scale!, packed_unscale!

"""
    PackedMatrix{R,V,Fmt}

Two-dimensional dense Hermitian square matrix in packed storage. Depending on Fmt, we only store the upper or lower triangle
in col-major vectorized or scaled vectorized form:
- `:U`: `PM[i, j] = PM[i + (j -1) * j ÷ 2]` for `1 ≤ i ≤ j ≤ n`
- `:L`: `PM[i, j] = PM[i + (j -1) * (2n - j) ÷ 2]` for `1 ≤ j ≤ i ≤ n`
- `:US`: `PM[i, j] = s(i, j) PM[i + (j -1) * j ÷ 2]` for `1 ≤ i ≤ j ≤ n`
- `:LS`: `PM[i, j] = s(i, j) PM[i + (j -1) * (2n - j) ÷ 2]` for `1 ≤ j ≤ i ≤ n`
where `n` is the side dimension of the packed matrix `PM`, `s(i, i) = 1`, and `s(i, j) = inv(sqrt(2))` for `i ≠ j`.

!!! warning
    The PackedMatrix can be effciently broadcast to other matrices, which works on the vector representation. It can also
    receive values from generic matrices by `copyto!` or broadcasting. Using it in a mixed chain of broadcastings of different
    type is not implemented and will potentially lead to logical errors (in the sense that the types will not match) or even
    segfaults (as the correct index mapping is not implemented).
"""
struct PackedMatrix{R,V<:AbstractVector{R},Fmt} <: AbstractMatrix{R}
    dim::Int
    data::V

    @doc """
        PackedMatrix(dim::Integer, data::AbstractVector{R}, type::Symbol=:U)

    Creates a packed matrix from already existing vectorized data. `type` must be one of `:U`, `:L`, `:US`, or `:LS`.
    """
    function PackedMatrix(dim::Integer, data::AbstractVector{R}, type::Symbol=:U) where {R}
        chkpacked(dim, data)
        type ∈ (:U, :L, :US, :LS) || error("Type must be one of :U, :L, :US, or :LS")
        return new{R,typeof(data),type}(dim, data)
    end

    @doc """
        PackedMatrix{R}(undef, dim, type::Symbol=:U)

    Creates a packed matrix with undefined data. `type` must be one of `:U`, `:L`, `:US`, or `:LS`.
    """
    function PackedMatrix{R}(::UndefInitializer, dim::Integer, type::Symbol=:U) where {R}
        type ∈ (:U, :L, :US, :LS) || error("Type must be one of :U, :L, :US, or :LS")
        return new{R,Vector{R},type}(dim, Vector{R}(undef, packedsize(dim)))
    end
end

function PackedMatrix(P::PackedMatrix, format::Symbol=packed_format(P))
    packed_isupper(P) == packed_isupper(format) || error("Changing the storage direction is not supported at the moment")
    if packed_isscaled(format)
        return packed_scale!(copy(P))
    else
        return packed_unscale!(copy(P))
    end
end

Base.size(P::PackedMatrix) = (P.dim, P.dim)
Base.eltype(::PackedMatrix{R}) where {R} = R
Base.require_one_based_indexing(P::PackedMatrix) = Base.require_one_based_indexing(P.data)

"""
    PackedMatrixUpper{R,V}

This is a union type for scaled and unscaled packed matrices with upper triangular storage.
"""
const PackedMatrixUpper{R,V} = Union{PackedMatrix{R,V,:U},PackedMatrix{R,V,:US}}
"""
    PackedMatrixLower{R,V}

This is a union type for scaled and unscaled packed matrices with lower triangular storage.
"""
const PackedMatrixLower{R,V} = Union{PackedMatrix{R,V,:L},PackedMatrix{R,V,:LS}}
"""
    PackedMatrixUnscaled{R,V}

This is a union type for unscaled packed matrices with upper or lower triangular storage.
"""
const PackedMatrixUnscaled{R,V} = Union{PackedMatrix{R,V,:U},PackedMatrix{R,V,:L}}
"""
    PackedMatrixScaled{R,V}

This is a union type for scaled packed matrices with upper or lower triangular storage.
"""
const PackedMatrixScaled{R,V} = Union{PackedMatrix{R,V,:US},PackedMatrix{R,V,:LS}}

packed_format(::PackedMatrix{R,V,Fmt}) where {R,V,Fmt} = Fmt
"""
    packed_isupper(::PackedMatrix)
    packed_isupper(::Type{<:PackedMatrix})

Returns `true` iff the given packed matrix has upper triangular storage.

See also [`packed_islower`](@ref), [`packed_isscaled`](@ref).
"""
packed_isupper(::Union{U,Type{U}} where {U<:PackedMatrixUpper}) = true
packed_isupper(::Union{L,Type{L}} where {L<:PackedMatrixLower}) = false
packed_isupper(s::Symbol) = s == :U || s == :US
"""
    packed_islower(::PackedMatrix)

Returns `true` iff the given packed matrix has lower triangular storage.

See also [`packed_isupper`](@ref), [`packed_isscaled`](@ref).
"""
packed_islower(::Union{U,Type{U}} where {U<:PackedMatrixUpper}) = false
packed_islower(::Union{L,Type{L}} where {L<:PackedMatrixLower}) = true
packed_islower(s::Symbol) = s == :L || s == :LS
@doc raw"""
    packed_isscaled(::PackedMatrix)

Returns `true` iff the given packed matrix has scaled storage, i.e., if the off-diagonal elements are internally stored with a
scaling of ``\sqrt2``.

See also [`packed_isupper`](@ref), [`packed_islower`](@ref).
"""
packed_isscaled(::Union{U,Type{U}} where {U<:PackedMatrixUnscaled}) = false
packed_isscaled(::Union{S,Type{S}} where {S<:PackedMatrixScaled}) = true
packed_isscaled(s::Symbol) = s == :US || s == :LS
packed_ulchar(x) = packed_isupper(x) ? 'U' : 'L'

@doc raw"""
    packed_scale!(P::PackedMatrix)

Ensures that `P` is a scaled packed matrix by multiplying off-diagonals by ``\sqrt2`` if necessary. Returns a scaled packed
matrix. `P` itself should not be referenced any more, only the result of this function.

See also [`packed_unscale!`](@ref).
"""
packed_scale!(P::PackedMatrixScaled) = P
packed_scale!(P::PackedMatrixUnscaled{R}) where {R} =
    rmul_offdiags!(PackedMatrix(P.dim, P.data, packed_isupper(P) ? :US : :LS), sqrt(R(2)))
@doc raw"""
    packed_unscale!(P::PackedMatrix)

Ensures that `P` is an unscaled packed matrix by dividing off-diagonals by ``\sqrt2`` if necessary. Returns an unscaled packed
matrix. `P` itself should not be referenced any more, only the result of this function.

See also [`packed_scale!`](@ref).
"""
packed_unscale!(P::PackedMatrixScaled{R}) where {R} =
    rmul_offdiags!(PackedMatrix(P.dim, P.data, packed_isupper(P) ? :U : :L), sqrt(inv(R(2))))
packed_unscale!(P::PackedMatrixUnscaled) = P

"""
    P[idx]

Returns the value that is stored at the position `idx` in the vectorized (possibly scaled) representation of `P`.
"""
Base.@propagate_inbounds Base.getindex(P::PackedMatrix, idx) = P.data[idx]
"""
    P[row, col]

Returns the value that is stored in the given row and column of `P`. This corresponds to the value in the matrix, so even if
`P` is of a scaled type, this does not affect the result.
"""
Base.@propagate_inbounds Base.getindex(P::PackedMatrix{R,V,:U}, row, col) where {R,V} =
    row ≤ col ? P.data[@inbounds rowcol_to_vec(P, row, col)] : conj(P.data[@inbounds rowcol_to_vec(P, col, row)])
Base.@propagate_inbounds function Base.getindex(P::PackedMatrix{R,V,:US}, row, col) where {R,V}
    if row < col
        return sqrt(inv(R(2))) * P.data[@inbounds rowcol_to_vec(P, row, col)]
    elseif row == col
        return P.data[@inbounds rowcol_to_vec(P, row, col)]
    else
        return sqrt(inv(R(2))) * conj(P.data[@inbounds rowcol_to_vec(P, col, row)])
    end
end
Base.@propagate_inbounds Base.getindex(P::PackedMatrix{R,V,:L}, row, col) where {R,V} =
    col ≤ row ? P.data[@inbounds rowcol_to_vec(P, row, col)] : conj(P.data[@inbounds rowcol_to_vec(P, col, row)])
Base.@propagate_inbounds function Base.getindex(P::PackedMatrix{R,V,:LS}, row, col) where {R,V}
    if row > col
        return sqrt(inv(R(2))) * P.data[@inbounds rowcol_to_vec(P, row, col)]
    elseif row == col
        return P.data[@inbounds rowcol_to_vec(P, row, col)]
    else
        return sqrt(inv(R(2))) * conj(P.data[@inbounds rowcol_to_vec(P, col, row)])
    end
end
"""
    P[idx] = X

Sets the value that is stored at the position `idx` in the vectorized (possibly scaled) representation of `P`.
"""
Base.@propagate_inbounds Base.setindex!(P::PackedMatrix, X, idx::Union{Integer,LinearIndices}) = P.data[idx] = X
@doc raw"""
    P[row, col] = X

Sets the value that is stored in the given row and column of `P`. This corresponds to the value in the matrix, so if `P` is of
a scaled type, `X` will internally be multiplied by ``\sqrt2``.
"""
Base.@propagate_inbounds function Base.setindex!(P::PackedMatrix{R,V,:U}, X, row, col) where {R,V}
    if row ≤ col
        P.data[@inbounds rowcol_to_vec(P, row, col)] = X
    else
        P.data[@inbounds rowcol_to_vec(P, col, row)] = conj(X)
    end
    return X
end
Base.@propagate_inbounds function Base.setindex!(P::PackedMatrix{R,V,:US}, X, row, col) where {R,V}
    if row < col
        P.data[@inbounds rowcol_to_vec(P, row, col)] = sqrt(R(2)) * X
    elseif row == col
        P.data[@inbounds rowcol_to_vec(P, row, col)] = X
    else
        P.data[@inbounds rowcol_to_vec(P, col, row)] = sqrt(R(2)) * X
    end
    return X
end
Base.@propagate_inbounds function Base.setindex!(P::PackedMatrix{R,V,:L}, X, row, col) where {R,V}
    if row ≥ col
        P.data[@inbounds rowcol_to_vec(P, row, col)] = X
    else
        P.data[@inbounds rowcol_to_vec(P, col, row)] = conj(X)
    end
    return X
end
Base.@propagate_inbounds function Base.setindex!(P::PackedMatrix{R,V,:LS}, X, row, col) where {R,V}
    if row > col
        P.data[@inbounds rowcol_to_vec(P, row, col)] = sqrt(R(2)) * X
    elseif row == col
        P.data[@inbounds rowcol_to_vec(P, row, col)] = X
    else
        P.data[@inbounds rowcol_to_vec(P, col, row)] = sqrt(R(2)) * X
    end
    return X
end
Base.IndexStyle(::PackedMatrix) = IndexLinear()
Base.IteratorSize(::Type{<:PackedMatrix}) = Base.HasLength()
Base.iterate(P::PackedMatrix, args...) = iterate(P.data, args...)
Base.length(P::PackedMatrix) = length(P.data)
Base.collect(P::PackedMatrix) = collect(P.data)
Base.fill!(P::PackedMatrix{R}, x::R) where {R} = fill!(P.data, x)

Base.copy(P::PackedMatrix) = PackedMatrix(P.dim, copy(P.data), packed_format(P))
for cp in (:copy!, :copyto!)
    for Fmt in (:(:U), :(:L), :(:US), :(:LS)) # we need to be so specific, as there would be ambiguities with the copyto!s below
        @eval begin
            Base.@propagate_inbounds function Base.$cp(dst::PackedMatrix{R,V,$Fmt} where {R,V},
                                                       src::PackedMatrix{R,V,$Fmt} where {R,V})
                $cp(dst.data, src.data)
                return dst
            end
        end
    end
    @eval begin
        Base.@propagate_inbounds function Base.$cp(dst::PackedMatrix{R}, src::AbstractVector{R}) where {R}
            $cp(dst.data, src)
            return dst
        end
        Base.@propagate_inbounds Base.$cp(dst::AbstractVector{R}, src::PackedMatrix{R}) where {R} = $cp(dst, src.data)
    end
end
@inline function Base.copyto!(dst::PackedMatrix{R,V,:U}, src::AbstractMatrix{R}) where {R,V}
    # Usually, copyto! only requires dst to be large enough to hold src, but this would become tricky in the case of packed
    # matrices. We could simply add the remaining offset to the bottom at the end in order to always copy the upper left
    # part... But what happens if the src matrix would then break the symmetry?
    @boundscheck checksquare(src) == dst.dim || throw(DimensionMismatch("Matrices must have the same size"))
    j = 1
    for (i, col) in enumerate(eachcol(src))
        @inbounds copyto!(dst.data, j, col, 1, i)
        j += i
    end
    return dst
end
@inline function Base.copyto!(dst::PackedMatrix{R,V,:US}, src::AbstractMatrix{R}) where {R,V}
    @boundscheck checksquare(src) == dst.dim || throw(DimensionMismatch("Matrices must have the same size"))
    j = 1
    @inbounds for (i, col) in enumerate(eachcol(src))
        @views dst.data[j:j+i-2] .= sqrt(R(2)) .* col[1:i-1]
        dst.data[j+i-1] = col[i]
        j += i
    end
    return dst
end
@inline function Base.copyto!(dst::PackedMatrix{R,V,:L}, src::AbstractMatrix{R}) where {R,V}
    @boundscheck checksquare(src) == dst.dim || throw(DimensionMismatch("Matrices must have the same size"))
    j = 1
    l = dst.dim
    for (i, col) in enumerate(eachcol(src))
        @inbounds copyto!(dst.data, j, col, i, l)
        j += l
        l -= 1
    end
end
@inline function Base.copyto!(dst::PackedMatrix{R,V,:LS}, src::AbstractMatrix{R}) where {R,V}
    @boundscheck checksquare(src) == dst.dim || throw(DimensionMismatch("Matrices must have the same size"))
    j = 1
    l = dst.dim
    @inbounds for (i, col) in enumerate(eachcol(src))
        dst.data[j] = col[i]
        @views dst.data[j+1:j+l-1] .= sqrt(R(2)) .* col[i+1:i+l-1]
        j += l
        l -= 1
    end
end
@inline function Base.copy!(dst::PackedMatrix{R}, src::AbstractMatrix{R}) where {R}
    @boundscheck checkbounds(src, axes(dst)...)
    return copyto!(dst, src)
end

function Base.similar(P::PackedMatrix{R}, ::Type{T}=eltype(P), dims::NTuple{2,Int}=size(P);
    format::Symbol=packed_format(P)) where {R,T}
    ==(dims...) || error("Packed matrices must be square")
    dim = first(dims)
    return PackedMatrix(dim, similar(P.data, T, packedsize(dim)), format)
end

Base.convert(T::Type{<:Ptr}, P::PackedMatrix) = convert(T, P.data)
Base.unsafe_convert(T::Type{Ptr{R}}, P::PackedMatrix{R}) where {R} = Base.unsafe_convert(T, P.data)
Base.reshape(P::PackedMatrix, ::Val{1}) = P.data # controversial? But it allows taking views with linear indices appropriately.
"""
    vec(P::PackedMatrix)

Returns the vectorized data associated with `P`. Note that this returns the actual vector, not a copy.
"""
LinearAlgebra.vec(P::PackedMatrix) = P.data

"""
    Matrix{R}(::PackedMatrix{R}, viewtype=R isa Complex ? Hermitian : Symmetric) where {R}

Construct a dense matrix from a packed matrix. The parameter `symmetric` determines which kind of view is returned.
"""
function Base.Matrix{R}(P::PackedMatrixUnscaled{R}, viewtype=R isa Complex ? Hermitian : Symmetric) where {R}
    result = Matrix{R}(undef, P.dim, P.dim)
    tpttr!(packed_ulchar(P), P.data, result)
    return viewtype(result, packed_isupper(P) ? :U : :L)
end
function Base.Matrix{R}(P::PackedMatrix{R,V,:US}, viewtype=R isa Complex ? Hermitian : Symmetric) where {R,V}
    result = Matrix{R}(undef, P.dim, P.dim)
    tpttr!('U', P.data, result)
    for j in 2:P.dim
        @inbounds rmul!(@view(result[1:j-1, j]), sqrt(inv(R(2))))
    end
    return viewtype(result, :U)
end
function Base.Matrix{R}(P::PackedMatrix{R,V,:LS}, viewtype=R isa Complex ? Hermitian : Symmetric) where {R,V}
    result = Matrix{R}(undef, P.dim, P.dim)
    tpttr!('L', P.data, result)
    for j in 1:P.dim-1
        @inbounds rmul!(@view(result[j+1:end, j]), sqrt(inv(R(2))))
    end
    return viewtype(result, :L)
end
"""
    PackedMatrix(::Union{<:Hermitian,<:Symmetric})

Construct a packed matrix from a Hermitian or symmetric wrapper of any other matrix. Note that in case a symmetric wrapper is
used, the element type must be invariant under conjugation (but this is not checked)!
"""
function PackedMatrix(A::LinearAlgebra.HermOrSym{R,<:AbstractMatrix{R}}) where {R}
    result = PackedMatrix{R}(undef, size(A, 1), A.uplo == 'U' ? :U : :L)
    trttp!(A.uplo, parent(A), result.data)
    return result
end