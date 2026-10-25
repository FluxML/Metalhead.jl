using Metalhead, Test
using JSON: JSON
using SafeTensors: SafeTensors
using Metalhead.Timm: pil_resize, eval_transform, resolve_data_config, PIL_RESAMPLING

# Written by porting/dump_timm_transform.py
const FILE = joinpath(pkgdir(Metalhead), "test", "references",
    "timm_transforms.safetensors")
const META = SafeTensors.deserialize(FILE).metadata
const REF = Metalhead.Checkpoint(FILE)
const CASES = filter(startswith("transform/"), collect(keys(META)))

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
