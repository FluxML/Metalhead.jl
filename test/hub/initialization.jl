using Metalhead, Flux, Test
using Metalhead: Random, Checkpoint, torch_params, source_weights, model_source,
                 source_config, identify, lookup, model_args, weights_name,
                 init_layer, init_array, build_pretrained, weight_init_context

const FIXTURES = joinpath(pkgdir(Metalhead), "test", "fixtures")

@testset "pretrained precision and weights" begin
    # Cover Flux layers, custom parameters, normalization, stacked experts and tied weights.
    for name in ("resnet_basic", "convnext_linear_mlp", "vit_dinov2_reg", "resmlp",
        "gmlp", "efficientnet_cc_tf_same", "llama_options", "gemma3_tiny",
        "granitemoe_tiny", "qwen3_5_tiny", "sam3_tiny", "vae_flux2_options",
        "transformer_flux2_options")
        dir = joinpath(FIXTURES, name)
        source = model_source(dir, nothing)
        config = source_config(source)
        format, architecture = identify(config)
        spec = lookup(architecture)
        args = model_args(format, spec, config, (;))
        checkpoint = Checkpoint(source_weights(source, weights_name(format)))
        for T in (Float16, Flux.BFloat16, Float32, Float64)
            @testset "$name $T" begin
                # The old path is the reference for both precision and checkpoint values.
                expected = Metalhead.ELTYPE_CONVERTERS[T](spec.build(; args...))
                Metalhead.load_state!(expected, checkpoint; ignore = [Metalhead.DEFAULT_IGNORE;
                                                                      spec.ignore])
                rng = copy(Random.default_rng())
                actual = create_model(dir; eltype = T)
                @test rand() == rand(rng) # pretrained construction runs no random initializers
                ps, aliases = torch_params(actual)
                refs, ref_aliases = torch_params(expected)
                @test aliases == ref_aliases
                @test [p.key for p in ps] == [p.key for p in refs]
                @test [eltype(p.array) for p in ps] == [eltype(p.array) for p in refs]
                @test all(p.array == r.array for (p, r) in zip(ps, refs))
            end
        end
    end
end

@testset "random construction" begin
    dir = joinpath(FIXTURES, "llama_options")
    for T in (Float16, Flux.BFloat16, Float32, Float64)
        Random.seed!(366)
        expected = Metalhead.ELTYPE_CONVERTERS[T](create_model(dir; pretrained = false))
        Random.seed!(366)
        actual = create_model(dir; pretrained = false, eltype = T)
        ps, aliases = torch_params(actual)
        refs, ref_aliases = torch_params(expected)
        @test aliases == ref_aliases
        @test all(p.array == r.array for (p, r) in zip(ps, refs))
    end
end

@testset "replacement classifiers" begin
    for (name, key) in (("resnet_basic", "fc"), ("convnext_linear_mlp", "head"),
        ("efficientnet_cc_tf_same", "classifier"), ("gmlp", "head"))
        dir = joinpath(FIXTURES, name)
        reference = Dict(p.key => p.array for p in first(torch_params(create_model(dir))))
        for T in (Float16, Flux.BFloat16, Float32, Float64)
            model = create_model(dir; num_classes = 3, eltype = T)
            ps = first(torch_params(model))
            head = filter(p -> p.key == key || startswith(p.key, key * "."), ps)
            @test !isempty(head)
            @test all(all(isfinite, p.array) for p in head)
            for p in ps
                (p.key == key || startswith(p.key, key * ".")) && continue
                @test p.array == eltype(p.array).(reference[p.key])
            end
            if name != "gmlp" # gMLP deliberately initializes its classifier weight to zero
                weight = only(filter(p -> endswith(p.key, "weight") && ndims(p.array) == 2, head))
                @test any(!iszero, weight.array)
            end
        end
    end
end

allocation_model() = init_layer(Embedding, 2048 => 512)

@testset "allocation and context lifetime" begin
    # Warm the builder, then allow a small overhead above the final 2 MiB of half weights.
    # A Float32 temporary or a second half-precision copy exceeds this bound.
    for T in (Float16, Flux.BFloat16)
        build_pretrained(allocation_model, (;), T)
        bytes = @allocated build_pretrained(allocation_model, (;), T)
        @test bytes < 2048 * 512 * sizeof(T) + 100_000
    end
    @test_throws "construction failed" build_pretrained(() -> error("construction failed"), (;), Float16)
    @test weight_init_context() === nothing
    @test eltype(init_array(Flux.ones32, 2)) == Float32

    # Interleaved tasks must not change one another's construction type.
    ready, resume = Channel{Nothing}(1), Channel{Nothing}(1)
    task = @async build_pretrained((;) -> begin
            put!(ready, nothing)
            take!(resume)
            allocation_model()
        end, (;), Float16)
    take!(ready)
    @test eltype(build_pretrained(allocation_model, (;), Float64).weight) == Float64
    @test eltype(allocation_model().weight) == Float32
    put!(resume, nothing)
    @test eltype(fetch(task).weight) == Float16
end
