using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup
using Flux
using Metalhead: torch_params, load_state!, Checkpoint
using Metalhead.Transformers: UperNetHead, UperNetFCNHead, UperNetConvModule,
                              UperNetPyramidPoolingBlock, adaptive_avg_pool,
                              interpolate_bilinear, cat_channels, top_down

# The heads of an UperNetForSemanticSegmentation, without its backbone: the model of the
# fixtures, which take the feature maps of a backbone.
struct UperNetHeads{D, A}
    decode_head::D
    auxiliary_head::A
end
Flux.@layer UperNetHeads
function (m::UperNetHeads)(features)
    return (; logits = m.decode_head(features),
        auxiliary_logits = m.auxiliary_head(features))
end

# The heads of an UperNetConfig (a config.json) on maps of `in_channels`.
function upernet_heads(config, in_channels; eltype = Float32)
    get_(k) = (v = config[k]; v isa Vector ? Tuple(v) : v)
    kw(ks) = (; (Symbol(k) => get_(k) for k in ks if haskey(config, k))...)
    num_labels = get(config, "num_labels", length(get(config, "id2label", (1, 2))))
    heads = UperNetHeads(
        UperNetHead(in_channels; num_labels, kw(["hidden_size", "pool_scales"])...),
        UperNetFCNHead(in_channels; num_labels,
            kw(["auxiliary_in_channels", "auxiliary_channels", "auxiliary_num_convs",
                "auxiliary_concat_input"])...))
    return Metalhead.ELTYPE_CONVERTERS[eltype](heads)
end

function fixture_heads(dir; eltype = Float64)
    config = JSON.parsefile(joinpath(dir, "heads_config.json"))
    heads = upernet_heads(config, config["in_channels"]; eltype)
    load_state!(heads, Checkpoint(joinpath(dir, "model.safetensors")))
    return Flux.testmode!(heads)
end

const FIXTURES = ["upernet_heads", "upernet_heads_options"]

@testset "fixtures" begin
    for name in FIXTURES
        r = check_fixture(fixture(name); model = fixture_heads(fixture(name)))
        @test r.output <= 1e-8
        @test all(t -> !ismissing(t.err), r.taps)   # every module of torch is ported
        @test r.pass
    end
end

@testset "adaptive average pooling" begin
    # torch.nn.AdaptiveAvgPool2d(3)(torch.arange(5.).view(1, 1, 1, 5)): overlapping bins
    x = reshape(collect(0.0:4.0), 5, 1, 1, 1)
    @test vec(adaptive_avg_pool(x, 3)[:, 1, 1, 1]) ≈ [0.5, 2.0, 3.5]
    # torch.nn.AdaptiveAvgPool2d(4) of 3 positions repeats them
    x = reshape([1.0, 2.0, 4.0], 3, 1, 1, 1)
    @test vec(adaptive_avg_pool(x, 4)[:, 1, 1, 1]) ≈ [1.0, 1.5, 3.0, 4.0]
    x = rand(Float32, 7, 5, 3, 2)
    @test adaptive_avg_pool(x, 1) ≈ sum(x; dims = (1, 2)) ./ 35
    @test adaptive_avg_pool(x, 1) isa Array{Float32, 4}
    y = rand(Float32, 8, 6, 3, 2)   # bins that divide: Flux's AdaptiveMeanPool
    @test adaptive_avg_pool(y, 2) ≈ AdaptiveMeanPool((2, 2))(y)
end

@testset "interpolation" begin
    # torch: F.interpolate(tensor([[[[1., 2.]]]]), size = (1, 4), mode = "bilinear")
    x = reshape(Float32[1, 2], 2, 1, 1, 1)
    @test vec(interpolate_bilinear(x, (4, 1))) ≈ Float32[1, 1.25, 1.75, 2]
    @test interpolate_bilinear(x, (2, 1)) === x
end

@testset "module tree" begin
    head = UperNetHead([4, 6, 8]; hidden_size = 8, pool_scales = (1, 2), num_labels = 3)
    keys = Set(p.key for p in first(torch_params(head)))
    @test "psp_modules.1.1.conv.weight" in keys
    @test "psp_modules.1.1.batch_norm.running_var" in keys
    @test "lateral_convs.1.conv.weight" in keys && "fpn_convs.1.conv.weight" in keys
    @test !("lateral_convs.2.conv.weight" in keys)   # the top level goes through the PPM
    @test "bottleneck.conv.weight" in keys && "fpn_bottleneck.conv.weight" in keys
    @test "classifier.bias" in keys && !any(endswith(".conv.bias"), keys)
    @test size(head.bottleneck.conv.weight) == (3, 3, 8 + 2 * 8, 8)
    @test size(head.fpn_bottleneck.conv.weight) == (3, 3, 3 * 8, 8)
    aux = UperNetFCNHead([4, 6, 8]; auxiliary_channels = 5, auxiliary_num_convs = 2,
        auxiliary_concat_input = true, num_labels = 3)
    @test [p.key for p in first(torch_params(aux))][[1, 6, 11, end]] ==
          ["convs.0.conv.weight", "convs.1.conv.weight", "conv_cat.conv.weight",
        "classifier.bias"]
    @test size(aux.convs[1].conv.weight) == (3, 3, 8, 5)       # the third map, 1-based
    @test size(aux.conv_cat.conv.weight) == (3, 3, 8 + 5, 5)
    @test isempty(first(torch_params(UperNetFCNHead([4, 6, 8]; auxiliary_channels = 3,
        auxiliary_num_convs = 0, auxiliary_in_channels = 3, num_labels = 2)))[1:(end - 2)])

    # maps of any size, the coarsest smaller than the largest pool scale
    features = [rand(Float32, 17, 13, 4, 2), rand(Float32, 9, 7, 6, 2),
        rand(Float32, 3, 2, 8, 2)]
    head = Flux.testmode!(UperNetHead([4, 6, 8]; hidden_size = 8, num_labels = 3))
    @test size(head(features)) == (17, 13, 3, 2)
    @test size(Flux.testmode!(aux)(features)) == (3, 2, 3, 2)
    @test size(Flux.testmode!(UperNetHead([8]; hidden_size = 4, num_labels = 2))(
        features[3:3])) == (3, 2, 2, 2)

    @test_throws ArgumentError head(features[1:2])
    @test_throws ArgumentError UperNetHead(Int[]; num_labels = 2)
    @test_throws ArgumentError UperNetHead([4, 0]; num_labels = 2)
    @test_throws ArgumentError UperNetHead([4]; pool_scales = (0,), num_labels = 2)
    @test_throws ArgumentError UperNetFCNHead([4, 6]; num_labels = 2)   # no third map
    @test_throws ArgumentError UperNetFCNHead([4, 6, 8]; auxiliary_num_convs = -1,
        num_labels = 2)
    @test_throws ArgumentError UperNetFCNHead([4, 6, 8]; auxiliary_num_convs = 0,
        auxiliary_channels = 5, num_labels = 2)
end

@testset "type stability" begin
    # the merging of the levels (Flux's Conv itself is not inferred)
    maps = [rand(Float32, 6, 5, 4, 1), rand(Float32, 3, 3, 4, 1)]
    @test @inferred(top_down(maps)) isa Vector{Array{Float32, 4}}
    @test @inferred(cat_channels(maps[1:1])) isa Array{Float32, 4}
    @test @inferred(adaptive_avg_pool(maps[1], 3)) isa Array{Float32, 4}
end

@testset "gradient" begin
    for name in FIXTURES
        heads = Flux.trainmode!(fixture_heads(fixture(name); eltype = Float32))
        args, _ = fixture_inputs(Checkpoint(joinpath(fixture(name), "acts.safetensors")))
        features = [Float32.(f) for f in only(args)]
        @test check_gradient(heads, features)
        g = Flux.gradient(f -> sum(abs2, heads(f).logits), features)[1]
        @test all(x -> x !== nothing && all(isfinite, x), g)   # to every feature map
    end
end

if TEST_NETWORK
    @testset "openmmlab/upernet-convnext-tiny" begin
        file = reference("openmmlab--upernet-convnext-tiny.safetensors")
        ref = Checkpoint(file)
        meta = SafeTensors.deserialize(file).metadata
        repo, revision = meta["repo"], meta["revision"]
        config = JSON.parsefile(Metalhead.hf_download(repo, "config.json"; revision))
        features = [ref["features/$j"] for j in 1:4]
        heads = upernet_heads(config, [size(f, 3) for f in features])
        # the weights of both heads load strictly, and only the backbone's are left
        r = load_state!(heads,
            Checkpoint(Metalhead.hf_download(repo, "model.safetensors"; revision));
            ignore = [Metalhead.DEFAULT_IGNORE; r"^backbone\."])
        @test isempty(r.missing) && isempty(r.unexpected) && isempty(r.mismatched)
        y = Flux.testmode!(heads)(features)
        for k in (:logits, :auxiliary_logits)
            @test size(y[k]) == size(ref[String(k)])
            @test maximum(abs, y[k] .- ref[String(k)]) <= 1e-4
            @test relerr(y[k], ref[String(k)]) <= 1e-5
        end
    end
end
