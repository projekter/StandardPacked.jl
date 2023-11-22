# idea: allow the PackedMatrix to work either as a vector or as a matrix, depending on the broadcasting context.
# However, this doesn't seem to work in all cases, so instead of providing a partially buggy implementation, we default to
# always treating the PackedMatrix as equivalent to its vector in broadcasting.
# We still perform a check so that we don't combine incompatible formats.
Base.broadcastable(P::PackedMatrix) = PackedMatrixBroadcasting{typeof(P)}(P)
struct PackedMatrixBroadcasting{PM<:PackedMatrix} <: AbstractVector{eltype(PM)}
    data::PM
end
Base.size(PB::PackedMatrixBroadcasting) = size(PB.data.data)
Base.axes(PB::PackedMatrixBroadcasting) = axes(PB.data.data)
Base.getindex(PB::PackedMatrixBroadcasting, ind::Int) = PB.data.data[ind]
Base.setindex!(PB::PackedMatrixBroadcasting, val, ind::Int) = PB.data.data[ind] = val
#=@inline Broadcast.combine_axes(A::PackedMatrixBroadcasting, B) = try
    return Broadcast.broadcast_shape(axes(A), axes(B))
catch
    return Broadcast.broadcast_shape(axes(A.data), axes(B))
end
@inline Broadcast.combine_axes(A, B::PackedMatrixBroadcasting) = try
    return Broadcast.broadcast_shape(axes(A), axes(B))
catch
    return Broadcast.broadcast_shape(axes(A), axes(B.data))
end
@inline Broadcast.combine_axes(A::PackedMatrixBroadcasting, B::PackedMatrixBroadcasting) = Broadcast.broadcast_shape(axes(A), axes(B))=#

struct PackedMatrixStyle{Fmt} <: Broadcast.AbstractArrayStyle{1} end
PackedMatrixStyle{Fmt}(::Val{1}) where {Fmt} = PackedMatrixStyle{Fmt}()
PackedMatrixStyle{Fmt}(::Val{2}) where {Fmt} = error("Broadcasting a PackedMatrix will only work on the vectorized data")
#PackedMatrixStyle{Fmt}(::Val{2}) where {Fmt} = PackedMatrixGenericStyle{Fmt}()
Base.BroadcastStyle(::Type{<:Union{<:PackedMatrixBroadcasting{<:PackedMatrix{R,V,Fmt} where {R,V}},<:PackedMatrix{R,V,Fmt} where {R,V}}}) where {Fmt} =
    PackedMatrixStyle{Fmt}()
Base.similar(bc::Broadcast.Broadcasted{P}, ::Type{T}) where {T,Fmt,P<:PackedMatrixStyle{Fmt}} =
    PackedMatrix{T}(undef, (isqrt(1 + 8length(bc)) -1) ÷ 2, Fmt)
    #=similar(find_pm(bc).data, T)
find_pm(bc::Base.Broadcast.Broadcasted) = find_pm(bc.args)
find_pm(args::Tuple) = find_pm(find_pm(args[1]), Base.tail(args))
find_pm(E::Broadcast.Extruded{<:PackedMatrixBroadcasting}) = E.x
find_pm(x) = x
find_pm(::Tuple{}) = nothing
find_pm(P::PackedMatrixBroadcasting, ::Any) = P
find_pm(::Any, rest) = find_pm(rest)=#
@inline function Base.copyto!(dest::PackedMatrix{R,V,Fmt} where {R,V}, bc::Broadcast.Broadcasted{PackedMatrixStyle{Fmt}}) where {Fmt}
    axes(dest.data) == axes(bc) || Broadcast.throwdm(axes(dest.data), axes(bc))
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if bc.f === identity && bc.args isa Tuple{AbstractArray} # only a single input argument to broadcast!
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end
    bc′ = Broadcast.preprocess(dest, bc)
    # Performance may vary depending on whether `@inbounds` is placed outside the
    # for loop or not. (cf. https://github.com/JuliaLang/julia/issues/38086)
    @inbounds @simd for I in eachindex(bc′)
        dest[I] = bc′[I]
    end
    return dest
end
Base.Broadcast.materialize!(s::PackedMatrixStyle{Fmt}, dest::PackedMatrix{R,V,Fmt} where {R,V}, bc::Broadcast.Broadcasted{PackedMatrixStyle{Fmt}}) where {Fmt} =
    (Base.Broadcast.materialize!(s, dest.data, bc); return dest)
Base.BroadcastStyle(::PackedMatrixStyle{Fmt}, ::PackedMatrixStyle{Fmt}) where {Fmt} = PackedMatrixStyle{Fmt}()
Base.BroadcastStyle(::PackedMatrixStyle, ::PackedMatrixStyle) =
    error("Packed matrices with different formats cannot be combined")
#=struct PackedMatrixGenericStyle{Fmt} <: Broadcast.AbstractArrayStyle{2} end
PackedMatrixGenericStyle{Fmt}(::Val{1}) where {Fmt} = PackedMatrixStyle{Fmt}()
PackedMatrixGenericStyle{Fmt}(::Val{2}) where {Fmt} = PackedMatrixGenericStyle{Fmt}()
Base.BroadcastStyle(::PackedMatrixStyle{Fmt}, ::Broadcast.DefaultArrayStyle{2}) where {Fmt} = PackedMatrixGenericStyle{Fmt}()
Base.BroadcastStyle(::PackedMatrixGenericStyle{Fmt}, ::Broadcast.DefaultArrayStyle{1}) where {Fmt} = PackedMatrixStyle{Fmt}()
Base.similar(bc::Broadcast.Broadcasted{<:PackedMatrixGenericStyle}, ::Type{T}) where {T} = similar(Array{T}, axes(bc))
@inline function Base.copyto!(dest::PackedMatrix{R,V,Fmt} where {R,V}, bc::Broadcast.Broadcasted{PackedMatrixGenericStyle{Fmt}}) where {Fmt}
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    if bc.f === identity && bc.args isa Tuple{AbstractArray}
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end
    bc′ = Broadcast.preprocess(dest, bc)
    @inbounds @simd for I in eachindex(IndexCartesian(), bc′) # linear indices will be out of bounds!
        dest[I] = bc′[I]
    end
    return dest
end=#