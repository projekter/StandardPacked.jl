__precompile__()
module StandardPacked

using LinearAlgebra, SparseArrays

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" include("../ext/StandardPackedStaticArrays.jl")
    end
end

include("spmatrix.jl")
include("utils.jl")
include("lapack.jl")
include("linalg.jl")
include("broadcasting.jl")

end