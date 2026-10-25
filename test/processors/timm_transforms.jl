using Metalhead, Test
using JSON: JSON
using SafeTensors: SafeTensors
using Metalhead.Timm: pil_resize, eval_transform, resolve_data_config, PIL_RESAMPLING,
                      EvalTransform

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

# The metadata of a transform case, and the settings resolved from its `pretrained_cfg`
function case_config(key)
    m = JSON.parse(META[key])
    return m, resolve_data_config(m["pretrained_cfg"]; use_test_size = m["use_test_size"])
end

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

@testset "resolve_data_config" begin
    defaults = (input_size = (3, 224, 224), interpolation = "bicubic",
        mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), crop_pct = 0.875,
        crop_mode = "center")
    @test resolve_data_config(Dict()) == defaults
    @test resolve_data_config((;)) == defaults
    @test resolve_data_config((; interpolation = "", mean = (), crop_pct = 0)) == defaults
    cfg = (input_size = (3, 224, 224), test_input_size = (3, 288, 288), crop_pct = 0.875,
        test_crop_pct = 1.0)
    @test resolve_data_config(cfg).input_size == (3, 224, 224)
    @test resolve_data_config(cfg).crop_pct == 0.875
    @test resolve_data_config(cfg; use_test_size = true).input_size == (3, 288, 288)
    @test resolve_data_config(cfg; use_test_size = true).crop_pct == 1.0
    for key in CASES  # as timm resolves them
        m, config = case_config(key)
        expected = m["config"]
        @test config.input_size == Tuple(expected["input_size"])
        @test config.mean == Tuple(expected["mean"])
        @test config.std == Tuple(expected["std"])
        for k in (:interpolation, :crop_pct, :crop_mode)
            @test config[k] == expected[String(k)]
        end
    end
end

@testset "eval_transform" begin
    images = [REF["image/$i"] for i in 1:7]  # RGB, but for a grayscale and an RGBA image
    @test size.(images, 3) == [3, 3, 3, 3, 3, 1, 4]
    @test length(CASES) == 14
    for key in CASES
        m, config = case_config(key)
        input = key == "transform/realistic" ? large_image(m["large_image"]...) : images
        x = eval_transform(input; config...)
        @test eltype(x) == Float32
        @test x == REF[key]  # bit for bit
    end

    _, config = case_config("transform/defaults")
    expected = REF["transform/defaults"]
    @test eval_transform(images[1]; config...) == expected[:, :, :, 1:1]
    # reals from 0 to 1, integers, and gray as a matrix
    @test eval_transform(images[1] ./ 255; config...) == expected[:, :, :, 1:1]
    @test eval_transform(Float32.(images[1]) ./ 255; config...) == expected[:, :, :, 1:1]
    @test eval_transform(Int.(images[1]); config...) == expected[:, :, :, 1:1]
    @test eval_transform(images[6][:, :, 1]; config...) == expected[:, :, :, 6:6]
    @test eval_transform(images[[1, 6]]; config...) == expected[:, :, :, [1, 6]]
    @test eval_transform(images[6]; config...) ==
          eval_transform(cat(images[6], images[6]; dims = 3); config...)

    @test size(eval_transform(images[1]; input_size = 12)) == (12, 12, 3, 1)
    @test size(eval_transform(images[1]; input_size = (3, 12, 20))) == (20, 12, 3, 1)
    @test_throws "crop_mode = \"border\" is not supported" eval_transform(images[1];
        crop_mode = "border")
    @test_throws "interpolation = \"random\" is not supported" eval_transform(images[1];
        interpolation = "random")
    @test_throws "from 0 to 1" eval_transform(images[1] ./ 254)
    @test_throws "from 0 to 255" eval_transform(Int.(images[1]) .+ 1)
    @test_throws "channels" eval_transform(rand(UInt8, 8, 8, 5))
    @test_throws "mean" eval_transform(images[1]; mean = (0.5,))
end

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

settings(t) = Metalhead.settings(t)
show_text(t) = sprint(show, MIME("text/plain"), t)

@testset "EvalTransform" begin
    images = [REF["image/$i"] for i in 1:7]
    # each case through a config.json with its pretrained_cfg, as timm's create_model reads it
    for key in CASES
        m, config = case_config(key)
        input = key == "transform/realistic" ? large_image(m["large_image"]...) : images
        with_config(timm_config(m["pretrained_cfg"])) do dir
            t = @test_logs create_processor(dir; use_test_size = m["use_test_size"])
            @test t isa EvalTransform
            @test settings(t) == config
            @test t(input) == REF[key]  # bit for bit
        end
    end

    # the config.json of a validated checkpoint
    m, config = case_config("transform/realistic")
    t = @test_logs create_processor(CONVNEXT)
    @test settings(t) == config
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

    # by hand, over the defaults of resolve_data_config
    @test settings(EvalTransform()) == resolve_data_config(Dict())
    t = EvalTransform(; input_size = (3, 16, 16), crop_pct = 0.95, interpolation = "box")
    @test t(images) == REF["transform/box"]
    @test_throws "one value per RGB channel" Metalhead.check_processor(EvalTransform(;
        mean = (0.5,)))
end

@testset "create_processor" begin
    images = [REF["image/$i"] for i in 1:7]
    # overrides
    t = create_processor(CONVNEXT; crop_pct = 1, interpolation = "bilinear")
    @test t.crop_pct === 1.0 && t.interpolation == "bilinear"
    @test occursin(r"crop_pct = 1\.0 +override \(config\.json: 0\.875\)", show_text(t))
    @test occursin(r"interpolation = \"bilinear\" +override \(config\.json: \"bicubic\"\)",
        show_text(t))
    @test t(images[1]) == eval_transform(images[1]; settings(t)...)
    message = "EvalTransform has no setting `crop_pc` (EvalTransform has `crop_pct`)"
    @test_throws message create_processor(CONVNEXT; crop_pc = 1.0)
    message = "`num_classes` is not used by EvalTransform: describes the classifier"
    @test_throws message create_processor(CONVNEXT; num_classes = 10)
    message = "crop_mode = \"border\" is not supported (supported: \"center\", \"squash\")"
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
        ("mean", [0.5], "one value per RGB channel"))
        with_config(timm_config(Dict(key => value))) do dir
            @test_throws message create_processor(dir)
        end
    end
    # false values in Python are the defaults, as resolve_data_config takes them
    with_config(timm_config(Dict("interpolation" => "", "mean" => [], "crop_pct" => 0,
        "crop_mode" => nothing, "std" => nothing))) do dir
        t = @test_logs create_processor(dir)
        @test settings(t) == settings(EvalTransform())
        @test occursin(r"crop_pct = 0\.875 +class default", show_text(t))
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
