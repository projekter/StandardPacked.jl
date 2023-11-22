using Test, Documenter, PackedMatrices

@testset "PackedMatrices" begin
    @testset "Documentation" begin
        doctest(PackedMatrices)
    end
    include("PackedMatrices.jl")
end