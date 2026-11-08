using Metalhead, Test
include("../parity_setup.jl")
using .ParitySetup

@testset "fixtures" begin
    for name in ["vit_tiny", "vit_options", "vit_dinov2", "vit_dinov2_reg", "vit_dynamic",
        "vit_dynamic_reg", "vit_clip", "vit_siglip", "vit_siglip_gap", "vit_avgmax",
        "vit_max_nopos"]
        r = check_fixture(fixture(name))
        @test r.output <= 1e-8
        @test r.features <= 1e-8
        @test r.pass
    end
end

@testset "architectures" begin
    for arch in list_models()
        startswith(arch, "vit_") || continue
        @test check_param_counts(arch)
    end
end

@testset "unsupported options" begin
    @test_throws "not supported" create_model("vit_tiny_patch16_224"; qk_norm = true)
    @test_throws "not supported" create_model("vit_tiny_patch16_224"; global_pool = "prr")
    @test_throws "not supported" create_model("vit_tiny_patch16_224"; act_layer = "relu")
    @test_throws "not supported" create_model("vit_tiny_patch16_224"; dynamic_img_pad = true)
    @test_throws "class_token" create_model("vit_tiny_patch16_224"; class_token = false)
end

@testset "img_size from the pretrained cfg" begin
    dir = mktempdir()
    config = Dict("architecture" => "vit_small_patch14_dinov2", "num_classes" => 0,
        "model_args" => Dict("embed_dim" => 16, "depth" => 1, "num_heads" => 2),
        "pretrained_cfg" => Dict("input_size" => [3, 28, 42], "fixed_input_size" => true))
    write(joinpath(dir, "config.json"), JSON.json(config))
    @test create_model(dir; pretrained = false).grid_size == (2, 3)
    @test create_model(dir; pretrained = false, img_size = 14).grid_size == (1, 1)
end

# A timm repository holding `model`: its config.json and the tensors of its state_dict.
function save_repository(dir, architecture, model_args, model)
    config = Dict("architecture" => architecture, "model_args" => model_args)
    write(joinpath(dir, "config.json"), JSON.json(config))
    tensors = Dict{String, AbstractArray}()
    for p in first(Metalhead.torch_params(model))
        x = p.kind == :dense ? permutedims(p.array) :
            p.kind == :conv ? reverse(p.array; dims = (1, 2)) : p.array
        tensors[p.key] = permutedims(x, ndims(x):-1:1)
    end
    SafeTensors.serialize(joinpath(dir, "model.safetensors"), tensors)
    return dir
end

# timm's checkpoint_filter_fn resamples the position embedding of a checkpoint to the grid
# of the model; the same model with `dynamic_img_size` resamples it in the forward pass.
@testset "checkpoint at another image size" begin
    resample = Metalhead.Timm.resample_abs_pos_embed
    tiny = (; patch_size = 8, embed_dim = 16, depth = 2, num_heads = 2, num_classes = 3)
    for (arch, args, prefix, sizes) in [
        ("vit_tiny_patch16_224", (; tiny..., img_size = 32), 1, [48, 16, (24, 40)]),
        ("vit_small_patch14_reg4_dinov2",
            (; tiny..., patch_size = 7, img_size = 28, reg_tokens = 2, global_pool = "avg"),
            0, [42, 14, (21, 35)]),
        ("vit_base_patch16_siglip_gap_224",
            (; tiny..., img_size = 32, num_classes = 0), 0, [48, 16, (24, 40)])]
        mktempdir() do dir
            model = create_model(arch; pretrained = false, args...)
            model_args = Dict(string(k) => v for (k, v) in pairs(args))
            save_repository(dir, arch, model_args, model)
            saved = Metalhead.Checkpoint(joinpath(dir, "model.safetensors"))["pos_embed"]
            @test create_model(dir).pos_embed == saved == model.pos_embed
            dynamic = create_model(dir; dynamic_img_size = true)
            for img_size in sizes
                h, w = Metalhead.Timm.to_2tuple(img_size)
                m = create_model(dir; img_size)
                grid = (h ÷ args.patch_size, w ÷ args.patch_size)
                @test m.grid_size == grid
                @test m.pos_embed == resample(saved, grid; num_prefix_tokens = prefix)
                x = rand(Float32, w, h, 3, 2)
                @test m(x) ≈ dynamic(x)
            end
            # timm resamples half precision in Float32, and a Float64 model gets the same
            m = create_model(dir; img_size = 48)
            @test create_model(dir; img_size = 48, eltype = Float16).pos_embed ==
                  Float16.(m.pos_embed)
            @test create_model(dir; img_size = 48, eltype = Float64).pos_embed ==
                  Float64.(m.pos_embed)
        end
    end

    # a checkpoint of PyTorch
    dir = fixture("vit_tiny")
    x = rand(Float32, 48, 48, 3, 2)
    resampled = create_model(dir; img_size = 48)
    @test resampled(x) ≈ create_model(dir; dynamic_img_size = true)(x)

    # the position embedding of a size that matches isn't touched, nor are other tensors
    m = create_model(dir)
    params = first(Metalhead.torch_params(m))
    pos_embed = Metalhead.Checkpoint(joinpath(dir, "model.safetensors"))["pos_embed"]
    p = only(p for p in params if p.key == "pos_embed")
    @test Metalhead.checkpoint_tensor(m, p, pos_embed) === pos_embed
    p = only(p for p in params if p.key == "cls_token")
    @test Metalhead.checkpoint_tensor(m, p, pos_embed) === pos_embed

    # a checkpoint grid that isn't square is left to the strict loader, as timm 1.0.30 can't
    # resample it
    mktempdir() do dir
        args = (; tiny..., img_size = (16, 32))
        model = create_model("vit_tiny_patch16_224"; pretrained = false, args...)
        save_repository(dir, "vit_tiny_patch16_224",
            Dict(string(k) => v for (k, v) in pairs(args)), model)
        @test create_model(dir).grid_size == (2, 4)
        @test_throws "mismatched: pos_embed" create_model(dir; img_size = 48)
    end
end

# timm builds a LayerScale only `if init_values`, so 0 gives an identity like None.
@testset "init_values = 0 builds no layer scale" begin
    T = Metalhead.Timm
    arch = "vit_tiny_patch16_224"
    tiny = (; patch_size = 8, embed_dim = 16, depth = 2, num_heads = 2, num_classes = 3,
        img_size = 16)
    keys_of(m) = [p.key for p in first(Metalhead.torch_params(m))]

    for init_values in (0, 0.0)
        m = create_model(arch; pretrained = false, tiny..., init_values)
        @test all(b -> b.ls1 === identity && b.ls2 === identity, m.blocks.layers)
    end
    m = create_model(arch; pretrained = false, tiny..., init_values = 1e-5)
    @test all(b -> b.ls1 isa T.LayerScale && b.ls2 isa T.LayerScale, m.blocks.layers)
    @test all(==(1.0f-5), m.blocks.layers[1].ls1.gamma)

    # a checkpoint without layer scale loads into the model with `init_values = 0`
    plain = create_model(arch; pretrained = false, tiny..., init_values = nothing)
    @test !any(contains(r"\.ls[12]\."), keys_of(plain))
    mktempdir() do dir
        save_repository(dir, arch, Dict(string(k) => v for (k, v) in pairs(tiny)), plain)
        loaded = create_model(dir; init_values = 0)
        @test keys_of(loaded) == keys_of(plain)
        x = rand(Float32, 16, 16, 3, 2)
        @test loaded(x) == plain(x)
    end
end

# timm loads the SigLIP models non-strictly when `global_pool != "map"`, which drops the
# tensors of the attention pool; Metalhead ignores those only.
@testset "SigLIP without the attention pool" begin
    dir = fixture("vit_siglip")
    x = rand(Float32, 32, 32, 3, 2)
    m = create_model(dir; global_pool = "avg", fc_norm = false)
    @test m.attn_pool === nothing
    @test size(m(x)) == (10, 2)
    @test size(create_model(fixture("vit_siglip_gap"))(x)) == (16, 2)

    # a final norm that moved to fc_norm is a real mismatch, which timm hides
    err = sprint(showerror, try
        create_model(dir; global_pool = "avg")
    catch e
        e
    end)
    @test occursin("missing:    fc_norm.weight", err)
    @test occursin("unused:     norm.weight", err)
    @test !occursin("attn_pool", err)

    # the attention pool still has to load
    mktempdir() do tmp
        cp(joinpath(dir, "config.json"), joinpath(tmp, "config.json"))
        st = SafeTensors.deserialize(joinpath(dir, "model.safetensors"))
        SafeTensors.serialize(joinpath(tmp, "model.safetensors"),
            Dict(k => collect(v) for (k, v) in st if k != "attn_pool.q.weight"))
        @test_throws "missing:    attn_pool.q.weight" create_model(tmp)
        @test size(create_model(tmp; global_pool = "avg", fc_norm = false)(x)) == (10, 2)
    end

    # other ViTs have no attention pool to ignore
    mktempdir() do tmp
        tiny = fixture("vit_tiny")
        cp(joinpath(tiny, "config.json"), joinpath(tmp, "config.json"))
        st = SafeTensors.deserialize(joinpath(tiny, "model.safetensors"))
        SafeTensors.serialize(joinpath(tmp, "model.safetensors"),
            Dict("attn_pool.latent" => zeros(Float32, 1, 1, 16),
                (k => collect(v) for (k, v) in st)...))
        @test_throws "unused:     attn_pool.latent" create_model(tmp)
    end
end

@testset "forward_intermediates" begin
    m = fixture_model(fixture("vit_dinov2_reg"); eltype = Float32)
    x = rand(Float32, 28, 21, 3, 2)
    features, ys = forward_intermediates(m, x)
    @test features ≈ forward_features(m, x)
    @test size.(ys) == [(4, 3, 16, 2), (4, 3, 16, 2)]
    nlc = forward_intermediates(m, x; output_fmt = "NLC", intermediates_only = true)
    @test reshape(permutedims(ys[1], (3, 1, 2, 4)), 16, 12, 2) ≈ nlc[1]
    pairs = forward_intermediates(m, x; return_prefix_tokens = true, intermediates_only = true)
    @test first(pairs[2]) ≈ ys[2]
    @test size(last(pairs[2])) == (16, 3, 2)
    @test forward_intermediates(m, x; indices = 1, intermediates_only = true) ≈ ys[2:2]
    truncated, y1 = forward_intermediates(m, x; indices = [1], stop_early = true)
    @test y1 ≈ ys[1:1]
    @test truncated ≈ m.norm(m.blocks.layers[1](Metalhead.Timm._embed(m, x)))
    @test_throws "indices" forward_intermediates(m, x; indices = [3])
    @test_throws "output_fmt" forward_intermediates(m, x; output_fmt = "NHWC")
    @test feature_info(m) ==
          [(; num_chs = 16, reduction = 7, module_name = "blocks.$i") for i in 0:1]
    g = Metalhead.Flux.gradient(
        m -> sum(abs2, only(forward_intermediates(m, x; indices = [1],
            intermediates_only = true))), m)[1]
    @test all(isfinite, g.blocks.layers[1].attn.qkv.weight)
    @test g.blocks.layers[2] === nothing
end

@testset "dynamic image size" begin
    m = fixture_model(fixture("vit_dynamic"); eltype = Float32)
    for (w, h) in [(28, 28), (21, 35), (42, 14)]
        @test size(m(rand(Float32, w, h, 3, 1))) == (16, 1)
        @test size(only(forward_intermediates(m, rand(Float32, w, h, 3, 1);
            indices = 1, intermediates_only = true))) == (w ÷ 7, h ÷ 7, 16, 1)
    end
end

@testset "half precision" begin
    for name in ["vit_tiny", "vit_siglip"]
        @test check_half_precision(name)
    end
    m = fixture_model(fixture("vit_dynamic"); eltype = Float32)
    @test check_half_precision(m, rand(Float32, 21, 35, 3, 2))
    # half-precision position embeddings are resampled in Float32, as in timm
    posemb = Float16.(randn(Float32, 8, 1 + 4 * 4, 1))
    @test Metalhead.Timm.resample_abs_pos_embed(posemb, (3, 5)) ==
          Float16.(Metalhead.Timm.resample_abs_pos_embed(Float32.(posemb), (3, 5)))
end

@testset "gradient" begin
    for (name, x) in [("vit_options", rand(Float32, 32, 16, 2, 2)),
        ("vit_dinov2_reg", rand(Float32, 28, 21, 3, 2)),
        ("vit_dynamic", rand(Float32, 21, 35, 3, 2)),
        ("vit_clip", rand(Float32, 32, 32, 3, 2)),
        ("vit_siglip", rand(Float32, 32, 32, 3, 2))]
        @test check_gradient(fixture_model(fixture(name); eltype = Float32), x)
    end
end

TEST_NETWORK && @testset "references" begin
    for file in reference_files("vit_")
        meta=SafeTensors.deserialize(file).metadata
        ref=Metalhead.Checkpoint(file)
        model=create_model(meta["repo"]; pretrained = true, revision = meta["revision"])
        x=reference_input(Tuple(JSON.parse(meta["input_shape"])))
        @test relerr(model(x), ref["logits"]) <= 1e-4
        @test relerr(forward_head(model, forward_features(model, x); pre_logits = true), ref["pre_logits"]) <=
              1e-4
    end
end

# DINOv2 at 224 instead of 518: timm resamples the position embedding as it loads
TEST_NETWORK && @testset "DINOv2 at another image size" begin
    repo, revision = first(Metalhead.lookup("vit_small_patch14_dinov2").references)
    x = rand(Float32, 224, 224, 3, 1)
    resampled = create_model(repo; revision, img_size = 224)
    @test resampled.grid_size == (16, 16)
    @test resampled(x) ≈ create_model(repo; revision, dynamic_img_size = true)(x)
end
