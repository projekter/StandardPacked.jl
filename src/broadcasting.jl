# Allow the SPMatrix to work either as a vector or as a matrix, depending on the broadcasting context.
# Kind corresponds to the format. We can in general combine SPMatrices with the same Fmt; we may also combine SPMatrices with
# vectors, but all involved SPMatrices must then have the same format. We may also combine SPMatrices with other matrices;
# then, any mixing of formats is allowed, but no vector may be present.
struct SPBroadcasting{Kind} <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:SPMatrix{<:Any,<:Any,Fmt}}) where {Fmt} = SPBroadcasting{Fmt}()
function Base.similar(bc::Broadcast.Broadcasted{SPBroadcasting{Fmt}}, ::Type{ElType}) where {ElType,Fmt}
    if Fmt ∈ (:U, :US, :L, :LS)
        return SPMatrix{ElType}(undef, length(axes(bc)[1]), Fmt)
    else
        return similar(Array{ElType}, axes(bc))
    end
end
const SPBroadcastingPacked = Union{SPBroadcasting{:U},SPBroadcasting{:US},SPBroadcasting{:L},SPBroadcasting{:LS}}
const SPBroadcastingVector = Union{SPBroadcasting{:VectorU},SPBroadcasting{:VectorUS},
                                   SPBroadcasting{:VectorL},SPBroadcasting{:VectorLS}}
Base.BroadcastStyle(::SPBroadcastingPacked, ::Union{<:SPBroadcastingPacked,SPBroadcasting{:Matrix},
                                                    <:Broadcast.AbstractArrayStyle{2}}) = SPBroadcasting{:Matrix}()
Base.BroadcastStyle(bc::SPBroadcasting, ::Broadcast.AbstractArrayStyle{0}) = bc
# same to same has default implementation; the rest must be combined via matrices or is incompatible
for kind in (:U, :US, :L, :LS)
    vkind = QuoteNode(Symbol(:Vector, kind))
    kind = QuoteNode(kind)
    @eval begin
        Base.BroadcastStyle(::SPBroadcasting{$kind}, ::Union{SPBroadcasting{$vkind},<:Broadcast.AbstractArrayStyle{1}}) =
            SPBroadcasting{$vkind}()
        function Base.convert(::Type{Broadcast.Broadcasted{SPBroadcasting{$vkind}}},
                              bc::Broadcast.Broadcasted{SPBroadcasting{$kind},Axes,F,Args}) where {Axes,F,Args}
            axes′ = (Base.OneTo(packedsize(length(axes(bc)[1]))),)
            return Broadcast.Broadcasted{SPBroadcasting{$vkind},typeof(axes′),F,Args}(bc.f, bc.args, axes′)
        end
    end
end

if VERSION < v"1.10"
    _Broadcasted(bc::Broadcast.Broadcasted, axes) = Broadcast.Broadcasted(bc.f, bc.args, axes)
else
    _Broadcasted(bc::Broadcast.Broadcasted, axes) = Broadcast.Broadcasted(bc.style, bc.f, bc.args, axes)
end
_unwrap(p::SPMatrix) = p.data
_unwrap(p) = p
function Broadcast.instantiate(bc::Broadcast.Broadcasted{<:SPBroadcastingVector})
    args = _unwrap.(bc.args)
    if bc.axes isa Nothing
        axes = Broadcast.combine_axes(args...)
    else
        axes = bc.axes
        Broadcast.check_broadcast_axes(axes, args...)
    end
    return _Broadcasted(bc, axes)
end
@inline function Broadcast.materialize!(::SPBroadcastingVector, dest::SPMatrix, bc::Broadcast.Broadcasted{<:Any})
    return copyto!(dest, Broadcast.instantiate(_Broadcasted(bc, axes(dest.data))))
end

@inline function Base.copyto!(dest::SPMatrix{R,<:Any,Fmt}, bc::Broadcast.Broadcasted{SPBroadcasting{Fmt′}}) where {R,Fmt,Fmt′}
    if Fmt === Fmt′
        axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    elseif Fmt′ === Symbol(:Vector, Fmt)
        axes(dest.data) == axes(bc) || Broadcast.throwdm(axes(dest.data), axes(bc))
    else
        throw(MethodError(copyto!, (dest, bc)))
    end
    destdata = dest.data
    if bc.f === identity && bc.args isa Tuple{SPMatrix}
        copyto!(destdata, bc.args[1].data)
        return dest
    end
    bc′ = Broadcast.preprocess(dest, convert(Broadcast.Broadcasted{SPBroadcasting{Symbol(:Vector, Fmt)}}, bc))
    @inbounds @simd for I in eachindex(bc′)
        destdata[I] = bc′[I]
    end
    return dest
end

@inline function Base.copyto!(dest::SPMatrix{R,<:Any,Fmt},
    bc::Broadcast.Broadcasted{<:Union{SPBroadcasting{:Matrix},<:Broadcast.AbstractArrayStyle{2}}}) where {R,Fmt}
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    bc′ = Broadcast.preprocess(dest, bc)
    if packed_isscaled(dest)
        sqrt2 = sqrt(R(2))
    end
    # small performance improvement: only assign one triangle
    idx = 1
    @inbounds for j in 1:size(dest, 2)
        if Fmt === :U
            for i in 1:j
                dest[idx] = bc′[i, j]
                idx += 1
            end
        elseif Fmt === :US
            for i in 1:j-1
                dest[idx] = sqrt2 * bc′[i, j]
                idx += 1
            end
            dest[idx] = bc′[j, j]
            idx += 1
        elseif Fmt === :L
            for i in j:size(dest, 1)
                dest[idx] = bc′[i, j]
                idx += 1
            end
        elseif Fmt === :LS
            dest[idx] = bc′[j, j]
            idx += 1
            for i in j+1:size(dest, 1)
                dest[idx] = sqrt2 * bc′[i, j]
                idx += 1
            end
        else
            @assert(false)
        end
    end
    return dest
end