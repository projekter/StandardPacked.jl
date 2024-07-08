module StandardPackedStaticArrays

import StandardPacked
import StaticArrays

# We need to be extra careful here - StaticArrays's broadcasting has higher precedence, but will try to write n^2 elements to
# our linear index. But we can spell this out and only copy the upper triangle.
@generated function StaticArrays._broadcast!(f, ::StaticArrays.Size{newsize}, dest::StandardPacked.SPMatrix,
    s::Tuple{Vararg{StaticArrays.Size}}, a...) where {newsize}
    sizes = [sz.parameters[1] for sz in s.parameters]

    indices = CartesianIndices(newsize)
    ps = StandardPacked.packedsize(newsize[1])
    exprs_eval = similar(indices, Expr, ps)
    exprs_setindex = similar(indices, Expr, ps)
    cmp = StandardPacked.packed_isupper(dest) ? Base.:≤ : Base.:≥
    j = 1
    for current_ind ∈ indices
        if cmp(current_ind[1], current_ind[2])
            exprs_vals = (StaticArrays.broadcast_getindex(sz, i, current_ind) for (i, sz) in enumerate(sizes))
            symb_val_j = Symbol(:val_, j)
            exprs_eval[j] = :($symb_val_j = f($(exprs_vals...)))
            exprs_setindex[j] = :(dest[$j] = $symb_val_j)
            j += 1
        end
    end

    return quote
        Base.@_inline_meta
        $(Expr(:block, exprs_eval...))
        @inbounds $(Expr(:block, exprs_setindex...))
        return dest
    end
end

end