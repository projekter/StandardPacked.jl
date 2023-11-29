using Documenter
using PackedMatrices

makedocs(sitename="PackedMatrices.jl", modules=[PackedMatrices],
    format=Documenter.HTML(prettyurls=false),
    pages=[
        "index.md",
        "reference.md",
        "lapack.md"
    ])

deploydocs(repo="github.com/projekter/PackedMatrices.jl.git")