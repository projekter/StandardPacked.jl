using Test, Documenter, StandardPacked

@testset "StandardPacked" begin
    @testset "Documentation" begin
        doctest(StandardPacked)
    end
    include("StandardPacked.jl")
end