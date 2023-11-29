__precompile__()
module StandardPacked

using LinearAlgebra, SparseArrays

include("spmatrix.jl")
include("utils.jl")
include("lapack.jl")
include("linalg.jl")
include("broadcasting.jl")

end