# idea: allow the SPMatrix to work either as a vector or as a matrix, depending on the broadcasting context.
# However, this doesn't seem to work in all cases, so instead of providing a partially buggy implementation, we default to
# always treating the SPMatrix as equivalent to its vector in broadcasting.
# We still perform a check so that we don't combine incompatible formats.
Base.broadcastable(P::SPMatrix) = SPMatrixBroadcasting{typeof(P)}(P)
struct SPMatrixBroadcasting{PM<:SPMatrix} <: AbstractVector{eltype(PM)}
    data::PM
end
Base.size(PB::SPMatrixBroadcasting) = size(PB.data.data)
Base.axes(PB::SPMatrixBroadcasting) = axes(PB.data.data)
Base.getindex(PB::SPMatrixBroadcasting, ind::Int) = PB.data.data[ind]
Base.setindex!(PB::SPMatrixBroadcasting, val, ind::Int) = PB.data.data[ind] = val
#=@inline Broadcast.combine_axes(A::SPMatrixBroadcasting, B) = try
    return Broadcast.broadcast_shape(axes(A), axes(B))
catch
    return Broadcast.broadcast_shape(axes(A.data), axes(B))
end
@inline Broadcast.combine_axes(A, B::SPMatrixBroadcasting) = try
    return Broadcast.broadcast_shape(axes(A), axes(B))
catch
    return Broadcast.broadcast_shape(axes(A), axes(B.data))
end
@inline Broadcast.combine_axes(A::SPMatrixBroadcasting, B::SPMatrixBroadcasting) = Broadcast.broadcast_shape(axes(A), axes(B))=#

struct SPMatrixStyle{Fmt} <: Broadcast.AbstractArrayStyle{1} end
SPMatrixStyle{Fmt}(::Val{1}) where {Fmt} = SPMatrixStyle{Fmt}()
SPMatrixStyle{Fmt}(::Val{2}) where {Fmt} = error("Broadcasting an SPMatrix will only work on the vectorized data")
#SPMatrixStyle{Fmt}(::Val{2}) where {Fmt} = SPMatrixGenericStyle{Fmt}()
Base.BroadcastStyle(::Type{<:Union{<:SPMatrixBroadcasting{<:SPMatrix{R,V,Fmt} where {R,V}},<:SPMatrix{R,V,Fmt} where {R,V}}}) where {Fmt} =
    SPMatrixStyle{Fmt}()
Base.similar(bc::Broadcast.Broadcasted{P}, ::Type{T}) where {T,Fmt,P<:SPMatrixStyle{Fmt}} =
    SPMatrix{T}(undef, (isqrt(1 + 8length(bc)) -1) ÷ 2, Fmt)
    #=similar(find_pm(bc).data, T)
find_pm(bc::Base.Broadcast.Broadcasted) = find_pm(bc.args)
find_pm(args::Tuple) = find_pm(find_pm(args[1]), Base.tail(args))
find_pm(E::Broadcast.Extruded{<:SPMatrixBroadcasting}) = E.x
find_pm(x) = x
find_pm(::Tuple{}) = nothing
find_pm(P::SPMatrixBroadcasting, ::Any) = P
find_pm(::Any, rest) = find_pm(rest)=#
@inline function Base.copyto!(dest::SPMatrix{R,V,Fmt} where {R,V}, bc::Broadcast.Broadcasted{SPMatrixStyle{Fmt}}) where {Fmt}
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
Base.Broadcast.materialize!(s::SPMatrixStyle{Fmt}, dest::SPMatrix{R,V,Fmt} where {R,V}, bc::Broadcast.Broadcasted{SPMatrixStyle{Fmt}}) where {Fmt} =
    (Base.Broadcast.materialize!(s, dest.data, bc); return dest)
Base.BroadcastStyle(::SPMatrixStyle{Fmt}, ::SPMatrixStyle{Fmt}) where {Fmt} = SPMatrixStyle{Fmt}()
Base.BroadcastStyle(::SPMatrixStyle, ::SPMatrixStyle) =
    error("Packed matrices with different formats cannot be combined")
#=struct SPMatrixGenericStyle{Fmt} <: Broadcast.AbstractArrayStyle{2} end
SPMatrixGenericStyle{Fmt}(::Val{1}) where {Fmt} = SPMatrixStyle{Fmt}()
SPMatrixGenericStyle{Fmt}(::Val{2}) where {Fmt} = SPMatrixGenericStyle{Fmt}()
Base.BroadcastStyle(::SPMatrixStyle{Fmt}, ::Broadcast.DefaultArrayStyle{2}) where {Fmt} = SPMatrixGenericStyle{Fmt}()
Base.BroadcastStyle(::SPMatrixGenericStyle{Fmt}, ::Broadcast.DefaultArrayStyle{1}) where {Fmt} = SPMatrixStyle{Fmt}()
Base.similar(bc::Broadcast.Broadcasted{<:SPMatrixGenericStyle}, ::Type{T}) where {T} = similar(Array{T}, axes(bc))
@inline function Base.copyto!(dest::SPMatrix{R,V,Fmt} where {R,V}, bc::Broadcast.Broadcasted{SPMatrixGenericStyle{Fmt}}) where {Fmt}
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