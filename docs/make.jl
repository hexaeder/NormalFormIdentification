using NormalFormIdentification
using Documenter
using Literate

DocMeta.setdocmeta!(NormalFormIdentification, :DocTestSetup, :(using NormalFormIdentification); recursive=true)

# Generate examples using Literate
example_dir = joinpath(@__DIR__, "examples")
outdir = joinpath(@__DIR__, "src", "generated")
isdir(outdir) && rm(outdir, recursive=true)
mkpath(outdir)

# Process any .jl files in examples directory
if isdir(example_dir)
    for example in filter(contains(r".jl$"), readdir(example_dir, join=true))
        Literate.markdown(example, outdir)
        Literate.script(example, outdir; keep_comments=true)
    end
end

# Build kwargs with strict and warn-only variants
kwargs = (;
    modules=[NormalFormIdentification],
    authors="Hans Würfel <git@wuerfel.io> and contributors",
    sitename="NormalFormIdentification.jl",
    root=joinpath(pkgdir(NormalFormIdentification), "docs"),
    linkcheck=true, # checks if external links resolve
    pagesonly=true,
    format=Documenter.HTML(;
        canonical="https://hexaeder.github.io/NormalFormIdentification.jl",
        edit_link="main",
        assets=String[],
        ansicolor = true,
        size_threshold=1_000_000_000
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Droop Identification" => "generated/droop_identification.md",
            "IEEE 9-Bus Linearization" => "generated/ieee9bus.md",
        ]
    ],
    draft=false,
    warnonly=[:missing_docs],
)
kwargs_warnonly = (; kwargs..., warnonly=true)

# Build with strict mode first, fallback to warn-only on failure
if haskey(ENV,"GITHUB_ACTIONS")
    success = true
    thrown_ex = nothing
    try
        makedocs(; kwargs...)
    catch e
        @info "Strict doc build failed, try again with warnonly=true"
        global success = false
        global thrown_ex = e
        makedocs(; kwargs_warnonly...)
    end

    deploydocs(;
        repo="github.com/hexaeder/NormalFormIdentification.jl",
        devbranch="main",
        push_preview=true
    )

    success || throw(thrown_ex)
else # local build
    makedocs(; kwargs_warnonly...)
end
