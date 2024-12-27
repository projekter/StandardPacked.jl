export SPMatrix, SPMatrixUpper, SPMatrixLower, SPMatrixUnscaled, SPMatrixScaled, packed_isupper,
    packed_islower, packed_isscaled, packed_scale!, packed_unscale!

"""
    SPMatrix{R,V,Fmt}

Two-dimensional dense Hermitian square matrix in packed storage. Depending on Fmt, we only store the upper or lower triangle
in col-major vectorized or scaled vectorized form:
- `:U`: `PM[i, j] = PM[i + (j -1) * j ÷ 2]` for `1 ≤ i ≤ j ≤ n`
- `:L`: `PM[i, j] = PM[i + (j -1) * (2n - j) ÷ 2]` for `1 ≤ j ≤ i ≤ n`
- `:US`: `PM[i, j] = s(i, j) PM[i + (j -1) * j ÷ 2]` for `1 ≤ i ≤ j ≤ n`
- `:LS`: `PM[i, j] = s(i, j) PM[i + (j -1) * (2n - j) ÷ 2]` for `1 ≤ j ≤ i ≤ n`
where `n` is the side dimension of the packed matrix `PM`, `s(i, i) = 1`, and `s(i, j) = inv(sqrt(2))` for `i ≠ j`.

!!! warning
    The SPMatrix can be effciently broadcast to other matrices, which works on the vector representation. It can also
    receive values from generic matrices by `copyto!` or broadcasting. Using it in a mixed chain of broadcastings of different
    type is not implemented and will potentially lead to logical errors (in the sense that the types will not match) or even
    segfaults (as the correct index mapping is not implemented).
"""
struct SPMatrix{R,V<:AbstractVector{R},Fmt} <: AbstractMatrix{R}
    dim::Int
    data::V

    @doc """
        SPMatrix(dim::Integer, data::AbstractVector{R}, type::Symbol=:U)

    Creates a packed matrix from already existing vectorized data. `type` must be one of `:U`, `:L`, `:US`, or `:LS`.
    """
    function SPMatrix(dim::Integer, data::AbstractVector{R}, type::Symbol=:U) where {R}
        chkpacked(dim, data)
        type ∈ (:U, :L, :US, :LS) || error("Type must be one of :U, :L, :US, or :LS")
        return new{R,typeof(data),type}(dim, data)
    end

    @doc """
        SPMatrix{R}(undef, dim, type::Symbol=:U)

    Creates a packed matrix with undefined data. `type` must be one of `:U`, `:L`, `:US`, or `:LS`.
    """
    function SPMatrix{R}(::UndefInitializer, dim::Integer, type::Symbol=:U) where {R}
        type ∈ (:U, :L, :US, :LS) || error("Type must be one of :U, :L, :US, or :LS")
        return new{R,Vector{R},type}(dim, Vector{R}(undef, packedsize(dim)))
    end
end

"""
    SPMatrix(P::SPMatrix, format=packed_format(P))

Creates a copy of the packed matrix `P`, possibly changing the format from scaled to unscaled. Note that a change between lower
or upper representation is not allowed.
"""
function SPMatrix(P::SPMatrix, format::Symbol=packed_format(P))
    packed_isupper(P) == packed_isupper(format) || error("Changing the storage direction is not supported at the moment")
    if packed_isscaled(format)
        return packed_scale!(copy(P))
    else
        return packed_unscale!(copy(P))
    end
end

Base.size(P::SPMatrix) = (P.dim, P.dim)
Base.eltype(::SPMatrix{R}) where {R} = R
Base.require_one_based_indexing(P::SPMatrix) = Base.require_one_based_indexing(P.data)

"""
    SPMatrixUpper{R,V}

This is a union type for scaled and unscaled packed matrices with upper triangular storage.
"""
const SPMatrixUpper{R,V} = Union{SPMatrix{R,V,:U},SPMatrix{R,V,:US}}
"""
    SPMatrixLower{R,V}

This is a union type for scaled and unscaled packed matrices with lower triangular storage.
"""
const SPMatrixLower{R,V} = Union{SPMatrix{R,V,:L},SPMatrix{R,V,:LS}}
"""
    SPMatrixUnscaled{R,V}

This is a union type for unscaled packed matrices with upper or lower triangular storage.
"""
const SPMatrixUnscaled{R,V} = Union{SPMatrix{R,V,:U},SPMatrix{R,V,:L}}
"""
    SPMatrixScaled{R,V}

This is a union type for scaled packed matrices with upper or lower triangular storage.
"""
const SPMatrixScaled{R,V} = Union{SPMatrix{R,V,:US},SPMatrix{R,V,:LS}}

packed_format(::SPMatrix{R,V,Fmt}) where {R,V,Fmt} = Fmt
"""
    packed_isupper(::SPMatrix)
    packed_isupper(::Type{<:SPMatrix})

Returns `true` iff the given packed matrix has upper triangular storage.

See also [`packed_islower`](@ref), [`packed_isscaled`](@ref).
"""
packed_isupper(::Union{U,Type{U}} where {U<:SPMatrixUpper}) = true
packed_isupper(::Union{L,Type{L}} where {L<:SPMatrixLower}) = false
packed_isupper(s::Symbol) = s == :U || s == :US
"""
    packed_islower(::SPMatrix)

Returns `true` iff the given packed matrix has lower triangular storage.

See also [`packed_isupper`](@ref), [`packed_isscaled`](@ref).
"""
packed_islower(::Union{U,Type{U}} where {U<:SPMatrixUpper}) = false
packed_islower(::Union{L,Type{L}} where {L<:SPMatrixLower}) = true
packed_islower(s::Symbol) = s == :L || s == :LS
@doc raw"""
    packed_isscaled(::SPMatrix)

Returns `true` iff the given packed matrix has scaled storage, i.e., if the off-diagonal elements are internally stored with a
scaling of ``\sqrt2``.

See also [`packed_isupper`](@ref), [`packed_islower`](@ref).
"""
packed_isscaled(::Union{U,Type{U}} where {U<:SPMatrixUnscaled}) = false
packed_isscaled(::Union{S,Type{S}} where {S<:SPMatrixScaled}) = true
packed_isscaled(s::Symbol) = s == :US || s == :LS
packed_ulchar(x) = packed_isupper(x) ? 'U' : 'L'

@doc raw"""
    packed_scale!(P::SPMatrix)

Ensures that `P` is a scaled packed matrix by multiplying off-diagonals by ``\sqrt2`` if necessary. Returns a scaled packed
matrix. `P` itself should not be referenced any more, only the result of this function.

See also [`packed_unscale!`](@ref).
"""
packed_scale!(P::SPMatrixScaled) = P
packed_scale!(P::SPMatrixUnscaled{R}) where {R} =
    rmul_offdiags!(SPMatrix(P.dim, P.data, packed_isupper(P) ? :US : :LS), sqrt(R(2)))
@doc raw"""
    packed_unscale!(P::SPMatrix)

Ensures that `P` is an unscaled packed matrix by dividing off-diagonals by ``\sqrt2`` if necessary. Returns an unscaled packed
matrix. `P` itself should not be referenced any more, only the result of this function.

See also [`packed_scale!`](@ref).
"""
packed_unscale!(P::SPMatrixScaled{R}) where {R} =
    rmul_offdiags!(SPMatrix(P.dim, P.data, packed_isupper(P) ? :U : :L), sqrt(inv(R(2))))
packed_unscale!(P::SPMatrixUnscaled) = P

@inline function Base._to_linear_index(P::SPMatrix{<:Any,<:Any,Fmt}, row::Integer, col::Integer) where {Fmt}
    if (Fmt === :U || Fmt === :US ? Base.:≤ : Base.:≥)(row, col)
        return @inbounds rowcol_to_vec(P, row, col)
    else
        return @inbounds rowcol_to_vec(P, col, row)
    end
end

"""
    P[idx]

Returns the value that is stored at the position `idx` in the vectorized (possibly scaled) representation of `P`.
"""
Base.@propagate_inbounds Base.getindex(P::SPMatrix, idx) = P.data[idx]
"""
    P[row, col]

Returns the value that is stored in the given row and column of `P`. This corresponds to the value in the matrix, so even if
`P` is of a scaled type, this does not affect the result.
"""
Base.@propagate_inbounds Base.getindex(P::SPMatrix{R,V,:U}, row, col) where {R,V} =
    row ≤ col ? P.data[@inbounds rowcol_to_vec(P, row, col)] : conj(P.data[@inbounds rowcol_to_vec(P, col, row)])
Base.@propagate_inbounds function Base.getindex(P::SPMatrix{R,V,:US}, row, col) where {R,V}
    if row < col
        return sqrt(inv(R(2))) * P.data[@inbounds rowcol_to_vec(P, row, col)]
    elseif row == col
        return P.data[@inbounds rowcol_to_vec(P, row, col)]
    else
        return sqrt(inv(R(2))) * conj(P.data[@inbounds rowcol_to_vec(P, col, row)])
    end
end
Base.@propagate_inbounds Base.getindex(P::SPMatrix{R,V,:L}, row, col) where {R,V} =
    col ≤ row ? P.data[@inbounds rowcol_to_vec(P, row, col)] : conj(P.data[@inbounds rowcol_to_vec(P, col, row)])
Base.@propagate_inbounds function Base.getindex(P::SPMatrix{R,V,:LS}, row, col) where {R,V}
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
Base.@propagate_inbounds Base.setindex!(P::SPMatrix, X, idx::Union{Integer,LinearIndices}) = P.data[idx] = X
@doc raw"""
    P[row, col] = X

Sets the value that is stored in the given row and column of `P`. This corresponds to the value in the matrix, so if `P` is of
a scaled type, `X` will internally be multiplied by ``\sqrt2``.
"""
Base.@propagate_inbounds function Base.setindex!(P::SPMatrix{R,V,:U}, X, row, col) where {R,V}
    if row ≤ col
        P.data[@inbounds rowcol_to_vec(P, row, col)] = X
    else
        P.data[@inbounds rowcol_to_vec(P, col, row)] = conj(X)
    end
    return X
end
Base.@propagate_inbounds function Base.setindex!(P::SPMatrix{R,V,:US}, X, row, col) where {R,V}
    if row < col
        P.data[@inbounds rowcol_to_vec(P, row, col)] = sqrt(R(2)) * X
    elseif row == col
        P.data[@inbounds rowcol_to_vec(P, row, col)] = X
    else
        P.data[@inbounds rowcol_to_vec(P, col, row)] = sqrt(R(2)) * conj(X)
    end
    return X
end
Base.@propagate_inbounds function Base.setindex!(P::SPMatrix{R,V,:L}, X, row, col) where {R,V}
    if row ≥ col
        P.data[@inbounds rowcol_to_vec(P, row, col)] = X
    else
        P.data[@inbounds rowcol_to_vec(P, col, row)] = conj(X)
    end
    return X
end
Base.@propagate_inbounds function Base.setindex!(P::SPMatrix{R,V,:LS}, X, row, col) where {R,V}
    if row > col
        P.data[@inbounds rowcol_to_vec(P, row, col)] = sqrt(R(2)) * X
    elseif row == col
        P.data[@inbounds rowcol_to_vec(P, row, col)] = X
    else
        P.data[@inbounds rowcol_to_vec(P, col, row)] = sqrt(R(2)) * conj(X)
    end
    return X
end
Base.IndexStyle(::SPMatrix) = IndexLinear()
Base.IteratorSize(::Type{<:SPMatrix}) = Base.HasLength()
Base.iterate(P::SPMatrix, args...) = iterate(P.data, args...)
Base.length(P::SPMatrix) = length(P.data)
Base.collect(P::SPMatrix) = collect(P.data)
Base.fill!(P::SPMatrix{R}, x::R) where {R} = fill!(P.data, x)

Base.copy(P::SPMatrix) = SPMatrix(P.dim, copy(P.data), packed_format(P))
for cp in (:copy!, :copyto!)
    for Fmt in (:(:U), :(:L), :(:US), :(:LS)) # we need to be so specific, as there would be ambiguities with the copyto!s below
        @eval begin
            Base.@propagate_inbounds function Base.$cp(dst::SPMatrix{R,V,$Fmt} where {R,V},
                                                       src::SPMatrix{R,V,$Fmt} where {R,V})
                $cp(dst.data, src.data)
                return dst
            end
        end
    end
    @eval begin
        Base.@propagate_inbounds function Base.$cp(dst::SPMatrix{R}, src::AbstractVector{R}) where {R}
            $cp(dst.data, src)
            return dst
        end
        Base.@propagate_inbounds Base.$cp(dst::AbstractVector{R}, src::SPMatrix{R}) where {R} = $cp(dst, src.data)
    end
end
@inline function Base.copyto!(dst::SPMatrix{R,V,:U}, src::AbstractMatrix{R}) where {R,V}
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
@inline function Base.copyto!(dst::SPMatrix{R,V,:US}, src::AbstractMatrix{R}) where {R,V}
    @boundscheck checksquare(src) == dst.dim || throw(DimensionMismatch("Matrices must have the same size"))
    j = 1
    @inbounds for (i, col) in enumerate(eachcol(src))
        @views dst.data[j:j+i-2] .= sqrt(R(2)) .* col[1:i-1]
        dst.data[j+i-1] = col[i]
        j += i
    end
    return dst
end
@inline function Base.copyto!(dst::SPMatrix{R,V,:L}, src::AbstractMatrix{R}) where {R,V}
    @boundscheck checksquare(src) == dst.dim || throw(DimensionMismatch("Matrices must have the same size"))
    j = 1
    l = dst.dim
    for (i, col) in enumerate(eachcol(src))
        @inbounds copyto!(dst.data, j, col, i, l)
        j += l
        l -= 1
    end
    return dst
end
@inline function Base.copyto!(dst::SPMatrix{R,V,:LS}, src::AbstractMatrix{R}) where {R,V}
    @boundscheck checksquare(src) == dst.dim || throw(DimensionMismatch("Matrices must have the same size"))
    j = 1
    l = dst.dim
    @inbounds for (i, col) in enumerate(eachcol(src))
        dst.data[j] = col[i]
        @views dst.data[j+1:j+l-1] .= sqrt(R(2)) .* col[i+1:i+l-1]
        j += l
        l -= 1
    end
    return dst
end
@inline function Base.copy!(dst::SPMatrix{R}, src::AbstractMatrix{R}) where {R}
    @boundscheck checkbounds(src, axes(dst)...)
    return copyto!(dst, src)
end

function Base.similar(P::SPMatrix{R}, ::Type{T}=eltype(P), dims::NTuple{2,Int}=size(P);
    format::Symbol=packed_format(P)) where {R,T}
    ==(dims...) || error("Packed matrices must be square")
    dim = first(dims)
    return SPMatrix(dim, similar(P.data, T, packedsize(dim)), format)
end

Base.convert(T::Type{<:Ptr}, P::SPMatrix) = convert(T, P.data)
Base.unsafe_convert(T::Type{Ptr{R}}, P::SPMatrix{R}) where {R} = Base.unsafe_convert(T, P.data)
Base.reshape(P::SPMatrix, ::Val{1}) = P.data # controversial? But it allows taking views with linear indices appropriately.
function Base._reshape(parent::SPMatrix, dims::Dims)
    n = parent.dim^2
    prod(dims) == n || Base._throw_dmrs(n, "size", dims)
    return Base.__reshape((parent, IndexCartesian()), dims)
end
"""
    vec(P::SPMatrix)

Returns the vectorized data associated with `P`. Note that this returns the actual vector, not a copy.
"""
LinearAlgebra.vec(P::SPMatrix) = P.data

function Base.convert(::Type{Matrix{R}}, P::SPMatrixUnscaled{R}) where {R}
    result = Matrix{R}(undef, P.dim, P.dim)
    tpttr!(packed_ulchar(P), P.data, result)
    return result
end
function Base.convert(::Type{Matrix{R}}, P::SPMatrix{R,V,:US}) where {R,V}
    result = Matrix{R}(undef, P.dim, P.dim)
    tpttr!('U', P.data, result)
    for j in 2:P.dim
        @inbounds rmul!(@view(result[1:j-1, j]), sqrt(inv(R(2))))
    end
    return result
end
function Base.convert(::Type{Matrix{R}}, P::SPMatrix{R,V,:LS}) where {R,V}
    result = Matrix{R}(undef, P.dim, P.dim)
    tpttr!('L', P.data, result)
    for j in 1:P.dim-1
        @inbounds rmul!(@view(result[j+1:end, j]), sqrt(inv(R(2))))
    end
    return result
end
"""
    Matrix{R}(P::SPMatrix{R}, viewtype=R isa Complex ? Hermitian : Symmetric) where {R}

Construct a dense matrix from a packed matrix. The parameter `symmetric` determines which kind of view is returned. Use
`convert(Matrix{R}, SPMatrix{R})` to return the matrix itself.
"""
Base.Matrix{R}(P::SPMatrix{R}, viewtype=R <: Complex ? Hermitian : Symmetric) where {R} =
    viewtype(convert(Matrix{R}, P), packed_isupper(P) ? :U : :L)
"""
    SPMatrix(A::Union{<:Hermitian,<:Symmetric})

Construct a packed matrix from a Hermitian or symmetric wrapper of any other matrix. Note that in case a symmetric wrapper is
used, the element type must be invariant under conjugation (but this is not checked)!
"""
function SPMatrix(A::LinearAlgebra.HermOrSym{R,<:AbstractMatrix{R}}) where {R}
    result = SPMatrix{R}(undef, size(A, 1), A.uplo == 'U' ? :U : :L)
    trttp!(A.uplo, parent(A), result.data)
    return result
end