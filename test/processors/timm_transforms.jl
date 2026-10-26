using Metalhead, Test
using JSON: JSON
using SafeTensors: SafeTensors
using Metalhead.Timm: pil_resize, PIL_RESAMPLING, EvalTransform

# Written by porting/dump_timm_transform.py
const FILE = joinpath(pkgdir(Metalhead), "test", "references",
    "timm_transforms.safetensors")
const META = SafeTensors.deserialize(FILE).metadata
const REF = Metalhead.Checkpoint(FILE)
const CASES = filter(startswith("transform/"), collect(keys(META)))
const PROCESSORS = joinpath(pkgdir(Metalhead), "test", "fixtures", "processors")
# The config.json of timm/convnext_tiny.fb_in1k, the case "realistic"
const CONVNEXT = joinpath(PROCESSORS, "timm--convnext_tiny.fb_in1k")
const TEST_NETWORK = parse(Bool, get(ENV, "METALHEAD_TEST_NETWORK", "false"))

# A directory with the config.json `config`
function with_config(f, config)
    mktempdir() do dir
        write(joinpath(dir, "config.json"), JSON.json(config))
        return f(dir)
    end
end
function timm_config(pretrained_cfg)
    return Dict("architecture" => "resnet18", "num_classes" => 1000,
        "pretrained_cfg" => pretrained_cfg)
end

# The transform of a config.json with the `pretrained_cfg`, as `create_processor` makes it
function transform_of(pretrained_cfg; kwargs...)
    return with_config(dir -> create_processor(dir; kwargs...), timm_config(pretrained_cfg))
end

# The metadata of a transform case, and its transform
function case_transform(key)
    m = JSON.parse(META[key])
    t = @test_logs transform_of(m["pretrained_cfg"]; use_test_size = m["use_test_size"])
    return m, t
end

settings(t) = Metalhead.settings(t)
show_text(t) = sprint(show, MIME("text/plain"), t)

# `large_image` of porting/dump_timm_transform.py
function large_image(c, h, w)
    return [UInt8((2h + 3w + 50c + mod(h * w * c + 7h + 11w, 37) * 3 +
                   mod(h ÷ 16 + w ÷ 16, 2) * 60) % 256)
            for w in 0:(w - 1), h in 0:(h - 1), c in 0:(c - 1)]
end

@testset "Pillow's resize" begin
    n = count(k -> endswith(k, "/input"), keys(REF))
    @test n == 10
    for i in 1:n
        image = REF["resize/$i/input"]
        size = Tuple(JSON.parse(META["resize/$i"])["size"])
        for filter in PIL_RESAMPLING
            @test pil_resize(image, size, filter) == REF["resize/$i/$filter"]
        end
        for filter in (:bilinear, :bicubic)
            @test pil_resize(image, size, filter; torchvision = true) ==
                  REF["resize/$i/torchvision_$filter"]
        end
    end
    image = REF["resize/1/input"]
    @test pil_resize(image, (28, 20)) == image
    @test pil_resize(view(image, :, :, 1:2), (5, 4), :bilinear) ==
          pil_resize(image, (5, 4), :bilinear)[:, :, 1:2]
    @test_throws "not supported" pil_resize(image, (5, 4), :cubic)
    @test_throws "not supported" pil_resize(image, (5, 4), :lanczos; torchvision = true)
    @test_throws "positive" pil_resize(image, (0, 4))
end

@testset "settings, as timm resolves them" begin
    defaults = (input_size = (3, 224, 224), interpolation = "bicubic",
        mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), crop_pct = 0.875,
        crop_mode = "center")
    @test settings(EvalTransform()) == defaults
    @test settings(transform_of(Dict())) == defaults
    # false values in Python are the defaults
    falsy = Dict("interpolation" => "", "mean" => [], "crop_pct" => 0, "crop_mode" => "",
        "std" => nothing)
    @test settings(@test_logs transform_of(falsy)) == defaults
    cfg = Dict("input_size" => [3, 224, 224], "test_input_size" => [3, 288, 288],
        "crop_pct" => 0.875, "test_crop_pct" => 1.0)
    @test transform_of(cfg).input_size == (3, 224, 224)
    @test transform_of(cfg).crop_pct == 0.875
    @test transform_of(cfg; use_test_size = true).input_size == (3, 288, 288)
    @test transform_of(cfg; use_test_size = true).crop_pct == 1.0
    for key in CASES  # as timm resolves them
        m, t = case_transform(key)
        expected = m["config"]
        @test t.input_size == Tuple(expected["input_size"])
        @test t.mean == Tuple(expected["mean"])
        @test t.std == Tuple(expected["std"])
        for k in (:interpolation, :crop_pct, :crop_mode)
            @test getfield(t, k) == expected[String(k)]
        end
    end
end

@testset "transform" begin
    images = [REF["image/$i"] for i in 1:7]  # RGB, but for a grayscale and an RGBA image
    @test size.(images, 3) == [3, 3, 3, 3, 3, 1, 4]
    @test length(CASES) == 14
    for key in CASES
        m, t = case_transform(key)
        input = key == "transform/realistic" ? large_image(m["large_image"]...) : images
        x = t(input)
        @test eltype(x) == Float32
        @test x == REF[key]  # bit for bit
    end

    _, t = case_transform("transform/defaults")
    expected = REF["transform/defaults"]
    @test t(images[1]) == expected[:, :, :, 1:1]
    # reals from 0 to 1, integers, and gray as a matrix
    @test t(images[1] ./ 255) == expected[:, :, :, 1:1]
    @test t(Float32.(images[1]) ./ 255) == expected[:, :, :, 1:1]
    @test t(Int.(images[1])) == expected[:, :, :, 1:1]
    @test t(images[6][:, :, 1]) == expected[:, :, :, 6:6]
    @test t(images[[1, 6]]) == expected[:, :, :, [1, 6]]
    @test t(images[6]) == t(cat(images[6], images[6]; dims = 3))

    @test size(EvalTransform(; input_size = (3, 12, 20))(images[1])) == (20, 12, 3, 1)
    @test_throws "from 0 to 1" t(images[1] ./ 254)
    @test_throws "from 0 to 255" t(Int.(images[1]) .+ 1)
    @test_throws "channels" t(rand(UInt8, 8, 8, 5))
end

@testset "EvalTransform" begin
    images = [REF["image/$i"] for i in 1:7]
    # the config.json of a validated checkpoint
    m, from_cfg = case_transform("transform/realistic")
    t = @test_logs create_processor(CONVNEXT)
    @test t isa EvalTransform
    @test settings(t) == settings(from_cfg)
    image = large_image(m["large_image"]...)
    @test t(image) == REF["transform/realistic"]
    @test t([image, image]) == cat(REF["transform/realistic"], REF["transform/realistic"];
        dims = 4)
    @test occursin(r"^EvalTransform\n  input_size = \(3, 224, 224\) +config\.json\n",
        show_text(t))
    @test occursin(r"crop_mode = \"center\" +config\.json$", show_text(t))
    @test sprint(show, t) ==
          "EvalTransform(input_size = (3, 224, 224), interpolation = \"bicubic\", \
           mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), crop_pct = 0.875, \
           crop_mode = \"center\")"

    # by hand, over the defaults
    t = EvalTransform(; input_size = (3, 16, 16), crop_pct = 0.95, interpolation = "box")
    @test t(images) == REF["transform/box"]
    # checked as it is made
    @test_throws "`crop_mode = \"border\"` is not supported" EvalTransform(;
        crop_mode = "border")
    @test_throws "`interpolation = \"random\"` is not supported" EvalTransform(;
        interpolation = "random")
    @test_throws "is not supported" EvalTransform(; settings(t)..., crop_mode = "border",
        t.provenance)
end

@testset "forms of the settings" begin
    images = [REF["image/$i"] for i in 1:7]
    # as timm's create_transform takes them
    t = EvalTransform(; input_size = 16, interpolation = :bilinear, crop_pct = 0.95,
        mean = 0.5, std = (0.5,))
    @test settings(t) == (input_size = (3, 16, 16), interpolation = "bilinear",
        mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5), crop_pct = 0.95, crop_mode = "center")
    @test t(images) == REF["transform/bilinear_095_inception"]
    @test EvalTransform(; input_size = (12, 20)).input_size == (3, 12, 20)
    @test EvalTransform(; input_size = [3, 12, 20]).input_size == (3, 12, 20)
    @test EvalTransform(; crop_mode = :squash).crop_mode == "squash"
    @test EvalTransform(; crop_pct = nothing, crop_mode = nothing) == EvalTransform()
    @test EvalTransform(; crop_pct = 0).crop_pct == 0.875
    # as overrides
    t = create_processor(CONVNEXT; input_size = 16, interpolation = :bilinear,
        crop_pct = 0.95, mean = 0.5, std = 0.5)
    @test t(images) == REF["transform/bilinear_095_inception"]
    @test occursin(r"input_size = \(3, 16, 16\) +override \(config\.json: \(3, 224, 224\)\)",
        show_text(t))
    @test create_processor(CONVNEXT; crop_pct = nothing).crop_pct == 0.875
    # one value in a configuration, which torchvision's `Normalize` broadcasts
    @test transform_of(Dict("mean" => [0.5], "std" => [0.25])).std == (0.25, 0.25, 0.25)
    # others are errors that name the form of the setting
    for (kwargs, message) in (
        (; input_size = 2.5) => "`input_size` is `(channels, height, width)`, `(height, width)` or an integer",
        (; input_size = (3, 0, 16)) => "`input_size` is `(channels, height, width)`",
        (; mean = (0.5, 0.5)) => "`mean` has one value per RGB channel, or one for all",
        (; std = "0.5") => "`std` has one value per RGB channel",
        (; interpolation = 2) => "`interpolation` is a string or a symbol",
        (; crop_mode = 1) => "`crop_mode` is a string or a symbol",
        (; crop_pct = -1) => "`crop_pct` is a positive number",
        (; crop_pct = "0.9") => "`crop_pct = \"0.9\"`: `crop_pct` is a positive number")
        @test_throws message EvalTransform(; kwargs...)
        @test_throws message create_processor(CONVNEXT; kwargs...)
    end
end

@testset "create_processor" begin
    images = [REF["image/$i"] for i in 1:7]
    # overrides
    t = create_processor(CONVNEXT; crop_pct = 1, interpolation = "bilinear")
    @test t.crop_pct === 1.0 && t.interpolation == "bilinear"
    @test occursin(r"crop_pct = 1\.0 +override \(config\.json: 0\.875\)", show_text(t))
    @test occursin(r"interpolation = \"bilinear\" +override \(config\.json: \"bicubic\"\)",
        show_text(t))
    @test t(images[1]) == EvalTransform(; settings(t)...)(images[1])
    message = "EvalTransform has no setting `crop_pc` (EvalTransform has `crop_pct`)"
    @test_throws message create_processor(CONVNEXT; crop_pc = 1.0)
    message = "`num_classes` is not used by EvalTransform: describes the classifier"
    @test_throws message create_processor(CONVNEXT; num_classes = 10)
    message = "`crop_mode = \"border\"` is not supported (supported: \"center\", \"squash\")"
    @test_throws message create_processor(CONVNEXT; crop_mode = "border")

    # use_test_size: test_input_size and test_crop_pct, where the configuration has them
    t = create_processor(CONVNEXT; use_test_size = true)
    @test t.input_size == (3, 288, 288) && t.crop_pct == 1.0
    @test occursin(r"input_size = \(3, 288, 288\) +config\.json \(`test_input_size`\)",
        show_text(t))
    t = create_processor(CONVNEXT; use_test_size = true, crop_pct = 0.9)
    @test t.input_size == (3, 288, 288) && t.crop_pct == 0.9
    @test occursin(r"crop_pct = 0\.9 +override \(config\.json \(`test_crop_pct`\): 1\.0\)",
        show_text(t))
    with_config(timm_config(Dict("input_size" => [3, 16, 16],
        "test_input_size" => [3, 20, 20], "crop_pct" => 0.95))) do dir
        t = create_processor(dir; use_test_size = true)
        @test t.input_size == (3, 20, 20) && t.crop_pct == 0.95
        @test create_processor(dir).input_size == (3, 16, 16)
    end
    sam3 = joinpath(PROCESSORS, "unsloth--sam3.1")
    message = "no component of Sam3Processor has a setting `use_test_size`"
    @test_throws message create_processor(sam3; use_test_size = true)

    # strict
    with_config(timm_config(Dict("input_size" => [3, 16, 16], "timm_test_key" => 1))) do dir
        t = @test_logs (:warn,
            "`timm_test_key` in config.json is not a setting of EvalTransform; it is ignored") create_processor(dir)
        @test t.input_size == (3, 16, 16)
    end
    for (key, value, message) in (("crop_mode", "border", "crop_mode = \"border\""),
        ("interpolation", "random", "interpolation = \"random\""),
        ("mean", [0.5, 0.5], "one value per RGB channel"))
        with_config(timm_config(Dict(key => value))) do dir
            @test_throws message create_processor(dir)
            # and the file that sets it
            @test_throws "`$key` is set in config.json" create_processor(dir)
        end
    end
    # false values in Python are the defaults, from the class
    with_config(timm_config(Dict("crop_pct" => 0))) do dir
        @test occursin(r"crop_pct = 0\.875 +class default", show_text(create_processor(dir)))
    end
    # timm's former config.json: the pretrained_cfg with the architecture
    with_config(Dict("architecture" => "resnet18", "num_features" => 512,
        "labels" => ["a", "b"], "num_classes" => 2, "input_size" => [3, 16, 16],
        "interpolation" => "bilinear")) do dir
        t = @test_logs create_processor(dir)
        @test t.input_size == (3, 16, 16) && t.interpolation == "bilinear"
    end
    # the processor of an architecture that isn't ported
    beit = Dict("architecture" => "beit_base_patch16_224",
        "pretrained_cfg" => Dict("crop_pct" => 0.9))
    with_config(beit) do dir
        @test create_processor(dir).crop_pct == 0.9
    end
end

relerr(x, y) = maximum(abs.(x .- y)) / maximum(abs.(y))

# The configurations of the validated checkpoints, as timm resolves them, and one checkpoint
# from a synthetic image to its logits
TEST_NETWORK && @testset "validated checkpoints" begin
    refs = [r for spec in values(Metalhead.REGISTRY)
            for r in spec.references
            if spec.format === Metalhead.Timm.TIMM]
    @test length(refs) == count(startswith("resolved/"), keys(META))
    for (repo, sha) in refs
        resolved = JSON.parse(META["resolved/$repo"])
        @test resolved["revision"] == sha
        for (use_test_size, key) in ((false, "config"), (true, "test_config"))
            t = @test_logs create_processor(repo; use_test_size)
            expected = resolved[key]
            @test t.input_size == Tuple(expected["input_size"])
            @test t.mean == Tuple(expected["mean"]) && t.std == Tuple(expected["std"])
            for k in (:interpolation, :crop_pct, :crop_mode)
                @test getfield(t, k) == expected[String(k)]
            end
        end
    end
    # a bare architecture, for its default checkpoint
    repo = first(first(Metalhead.REGISTRY["resnet50"].references))
    @test settings(create_processor("resnet50")) == settings(create_processor(repo))

    m = JSON.parse(META["transform/realistic"])
    model = create_model(m["repo"]; pretrained = true)
    x = create_processor(m["repo"])(large_image(m["large_image"]...))
    @test x == REF["transform/realistic"]
    logits = model(x)
    @test relerr(logits, REF["logits/realistic"]) <= 1.0e-4
    @test argmax(vec(logits)) == argmax(vec(REF["logits/realistic"]))
end
