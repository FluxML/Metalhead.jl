using Metalhead, Test
using Flux, ConcreteStructs

@concrete struct Block
    conv
    norm
    proj
    scale
end
Flux.@layer Block

@testset "torch keys from model structure" begin
    using Metalhead: torch_params

    m = Chain(
        Block(Conv((3, 3), 2 => 4), BatchNorm(4), Dense(4 => 4; bias = false), ones(Float32, 4)),
        [Dense(4 => 4), LayerNorm(4)],
        (a = Embedding(5 => 4), b = relu, c = SkipConnection(Dense(4 => 4), +)))
    keys = [p.key for p in first(torch_params(m))]
    @test keys == ["0.conv.weight", "0.conv.bias",
        "0.norm.weight", "0.norm.bias", "0.norm.running_mean", "0.norm.running_var",
        "0.proj.weight", "0.scale",
        "1.0.weight", "1.0.bias", "1.1.weight", "1.1.bias",
        "2.a.weight", "2.c.weight", "2.c.bias"]
    kinds = Dict(p.key => p.kind for p in first(torch_params(m)))
    @test kinds["0.conv.weight"] == :conv
    @test kinds["0.proj.weight"] == :dense
    @test kinds["2.a.weight"] == :plain

    @test first(first(torch_params(Dense(2 => 3); prefix = "head"))).key == "head.weight"

    shared = Dense(3 => 3)
    ps, aliases = torch_params((enc = shared, dec = shared))
    @test [p.key for p in ps] == ["enc.weight", "enc.bias"]
    @test aliases == ["dec.weight" => "enc.weight", "dec.bias" => "enc.bias"]

    @test_throws "no PyTorch mapping" torch_params(Chain(Flux.Scale(3)))
end

@testset "loading PyTorch-layout tensors" begin
    using Metalhead: Checkpoint, load_state!
    using Flux, SafeTensors

    # Tensors are written in PyTorch (row-major logical) order.
    torch_dense = Float32[1 2 3; 4 5 6]                  # Linear(3 => 2).weight, shape (out, in)
    torch_conv = reshape(Float32.(1:(2 * 1 * 2 * 3)), 2, 1, 2, 3)  # Conv2d(1 => 2, (2, 3)).weight
    torch_emb = Float32[1 2; 3 4; 5 6]                   # Embedding(3, 2).weight, shape (vocab, dim)
    dir = mktempdir()
    path = joinpath(dir, "model.safetensors")
    SafeTensors.serialize(path,
        Dict("fc.weight" => torch_dense, "fc.bias" => Float32[7, 8],
            "conv.weight" => torch_conv, "emb.weight" => torch_emb,
            "bn.num_batches_tracked" => Int64[3]))
    ckpt = Checkpoint(path)
    @test ckpt["fc.weight"] == permutedims(torch_dense)
    @test size(ckpt["conv.weight"]) == (3, 2, 1, 2)

    m = (fc = Dense(3 => 2), conv = Conv((3, 2), 1 => 2; bias = false),
        emb = Embedding(3 => 2))
    report = load_state!(m, ckpt)
    @test report.loaded == 4
    @test m.fc.weight == torch_dense
    @test m.emb.weight == permutedims(torch_emb)

    # PyTorch Conv2d is a cross-correlation over (H, W); Flux's Conv is a convolution over (W, H).
    x = reshape(Float32.(1:20), 5, 4, 1, 1)              # Julia (W, H, C, N)
    y = m.conv(x)
    torch_out(o, i, j) = sum(torch_conv[o, 1, a, b] * x[j + b - 1, i + a - 1, 1, 1]
    for a in 1:2, b in 1:3)
    @test y[:, :, 2, 1] ≈ [torch_out(2, i, j) for j in 1:3, i in 1:3]
end

@testset "strict loading errors" begin
    using Metalhead: Checkpoint, load_state!
    using Flux, SafeTensors

    dir = mktempdir()
    path = joinpath(dir, "model.safetensors")
    SafeTensors.serialize(path,
        Dict("fc.weigth" => zeros(Float32, 2, 3), "fc.bias" => zeros(Float32, 3),
            "extra" => zeros(Float32, 1)))
    m = (; fc = Dense(3 => 2))
    err = try
        load_state!(m, Checkpoint(path))
    catch e
        sprint(showerror, e)
    end
    @test occursin("missing:    fc.weight  (closest unused checkpoint key: fc.weigth)", err)
    @test occursin("unused:     extra", err)
    @test occursin("mismatched: fc.bias", err)

    report = load_state!(m, Checkpoint(path); strict = false)
    @test report.missing == ["fc.weight"]
    @test report.unexpected == ["extra", "fc.weigth"]

    # `skip` leaves the parameters of a submodule (e.g. a replaced classifier) untouched.
    SafeTensors.serialize(path,
        Dict("fc.weight" => ones(Float32, 2, 3), "fc.bias" => ones(Float32, 2),
            "head.weight" => ones(Float32, 5, 2), "head.bias" => ones(Float32, 5)))
    m = (fc = Dense(3 => 2), head = Dense(2 => 7))
    w = copy(m.head.weight)
    load_state!(m, Checkpoint(path); skip = "head")
    @test m.fc.weight == ones(Float32, 2, 3)
    @test m.head.weight == w
end

@testset "create_model" begin
    using Flux
    @test "resnet50" in list_models()
    m = create_model("resnet18"; num_classes = 7)
    @test size(m(rand(Float32, 64, 64, 3, 2))) == (7, 2)
    @test m.bn1.active === false
    @test size(forward_features(m, rand(Float32, 64, 64, 3, 1))) == (2, 2, 512, 1)
    @test eltype(create_model("resnet18"; eltype = Float64).fc.weight) == Float64
    mbf = create_model("resnet18"; eltype = Flux.BFloat16)  # run on GPUs only
    @test eltype(mbf.fc.weight) == Flux.BFloat16
    @test eltype(create_model("resnet18"; eltype = Float16)(rand(Float16, 64, 64, 3, 1))) ==
          Float16
    @test create_model("resnet18"; num_classes = 0).fc === identity
    @test_throws "neither a ported architecture" create_model("not_a_model")
    @test_throws "not supported" create_model("resnet18"; aa_layer = "blur")
    @test_throws MethodError create_model("resnet18"; not_a_kwarg = 1)
end
