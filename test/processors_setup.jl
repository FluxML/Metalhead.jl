# Helpers of the tests of test/processors/, included by them.
using Markdown: Markdown

"""
    documented_keys(T) -> Vector{Symbol}

The keys of the "Returns" tables of the docstring of the processor `T`: the first column,
in code, of its tables whose header begins with "Key". A test checks them against the keys
of `proc(...)` on the fixtures, so that the documented outputs can't drift.
"""
function documented_keys(T)
    found = Symbol[]
    visit(x) = nothing
    visit(md::Markdown.MD) = foreach(visit, md.content)
    function visit(t::Markdown.Table)
        startswith(Markdown.plain(Markdown.Paragraph(first(t.rows)[1])), "Key") || return
        for row in t.rows[2:end]
            push!(found, Symbol(only(c for c in row[1] if c isa Markdown.Code).code))
        end
    end
    visit(Base.Docs.doc(T))
    return found
end

"""
    check_documented_keys(T, outputs)

Test that every key of each NamedTuple of `outputs`, which processors of type `T` returned,
is in the "Returns" table of `T`'s docstring, and that every documented key is in one of
them.
"""
function check_documented_keys(T, outputs)
    documented = documented_keys(T)
    @test !isempty(documented)
    @test allunique(documented)
    for out in outputs
        @test issubset(keys(out), documented)
    end
    @test issubset(documented, union(Set{Symbol}(), keys.(outputs)...))
end
