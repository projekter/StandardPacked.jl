using Documenter
using PackedMatrices

makedocs(sitename="PackedMatrices.jl", modules=[PackedMatrices],
    format=Documenter.HTML(prettyurls=false),
    pages=[
        "index.md",
        "reference.md",
    ],
    remotes=nothing)