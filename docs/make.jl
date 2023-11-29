using Documenter
using StandardPacked

makedocs(sitename="StandardPacked.jl", modules=[StandardPacked],
    format=Documenter.HTML(prettyurls=false),
    pages=[
        "index.md",
        "reference.md",
        "lapack.md"
    ])

deploydocs(repo="github.com/projekter/StandardPacked.jl.git")