using Metalhead, Test
using ImageCore
using Metalhead: image_array, image_batch, image_matrix
include(joinpath(@__DIR__, "..", "data.jl"))
using .TestData: test_data

const PROCESSORS = joinpath(pkgdir(Metalhead), "test", "fixtures", "processors")
const TIMM = Metalhead.Checkpoint(test_data("references/timm_transforms.safetensors"))
const SAM3 = Metalhead.Checkpoint(test_data("references/processors/sam3.safetensors"))

# The image of colorants that an array (width, height, channels) of UInt8 stands for, a
# matrix (height, width) as FileIO loads it
function colorants(x::AbstractArray{UInt8, 3})
    raw = permutedims(reinterpret(N0f8, x), (3, 2, 1))
    size(x, 3) == 1 && return collect(colorview(Gray, raw[1, :, :]))
    return collect(colorview((nothing, GrayA, RGB, RGBA)[size(x, 3)], raw))
end
colorants(x::AbstractMatrix{UInt8}) = colorants(reshape(x, size(x)..., 1))

@testset "image_array" begin
    x = rand(UInt8, 7, 5, 4)
    rgba = colorants(x)
    @test rgba isa Matrix{RGBA{N0f8}} && size(rgba) == (5, 7)
    @test image_array(rgba) isa Array{UInt8, 3}
    @test image_array(rgba) == x
    @test image_array(RGB.(rgba)) == x[:, :, 1:3]
    @test image_array(rgba) == permutedims(rawview(channelview(rgba)), (3, 2, 1))
    # the channels in the order R, G, B and alpha, whatever the order of the fields
    for C in (BGR, XRGB, RGBX, RGB24)
        @test image_array(C.(rgba)) == x[:, :, 1:3]
    end
    for C in (ARGB, BGRA, ABGR, ARGB32)
        @test image_array(C.(rgba)) == x
    end
    grey = colorants(x[:, :, 1])
    @test grey isa Matrix{Gray{N0f8}}
    @test image_array(grey) == x[:, :, 1:1]
    @test image_array(Gray24.(grey)) == x[:, :, 1:1]
    @test image_array(colorants(x[:, :, 1:2])) == x[:, :, 1:2]
    @test image_array(AGray.(colorants(x[:, :, 1:2]))) == x[:, :, 1:2]
    # other components as reals from 0 to 1, Bool as 8 bits
    @test image_array(RGB{Float32}.(rgba)) == Float32.(reinterpret(N0f8, x[:, :, 1:3]))
    @test image_array(RGB{Float64}.(rgba)) isa Array{Float64, 3}
    @test image_array(RGB{N0f16}.(rgba)) ≈ x[:, :, 1:3] ./ 255
    @test image_array(Gray.([true false; false true])) == reshape([0xff 0; 0 0xff], 2, 2, 1)
    @test_throws "convert them to RGB" image_array(HSV.(RGB.(rgba)))
    @test image_batch([rgba, rgba]) == [x, x]
    @test image_batch(rgba) == [x]
end

@testset "image_matrix" begin
    # the inverse of image_array, for images in the layout of processors and pipelines
    x = rand(UInt8, 7, 5, 4)
    for c in 1:4
        img = image_matrix(x[:, :, 1:c])
        @test img == colorants(x[:, :, 1:c])
        @test image_array(img) == x[:, :, 1:c]
    end
    @test image_matrix(x[:, :, 1:3]) isa AbstractMatrix{RGB{N0f8}}
    @test image_matrix(x[:, :, 1]) == colorants(x[:, :, 1])
    @test image_matrix(Int.(x[:, :, 1:3])) == colorants(x[:, :, 1:3])
    @test_throws "from 0 to 255" image_matrix(fill(256, 2, 2, 3))
    # reals from 0 to 1, in Float32, as a pipeline generates them in any precision
    y = rand(Float32, 6, 4, 3, 2)
    imgs = image_matrix(y)
    @test length(imgs) == 2 && imgs[2] isa AbstractMatrix{RGB{Float32}}
    @test size(imgs[2]) == (4, 6) && imgs[2][3, 5] == RGB(y[5, 3, :, 2]...)
    @test image_array(imgs[2]) == y[:, :, :, 2]
    @test image_matrix(Float16.(y[:, :, :, 1])) isa AbstractMatrix{RGB{Float32}}
    # a mask of SAM 3 (width, height)
    mask = rand(Bool, 6, 4)
    @test image_matrix(mask) == Gray.(permutedims(mask))
    @test_throws "1 to 4 channels" image_matrix(rand(Float32, 2, 2, 5))
    @test_throws "takes an array of reals" image_matrix([RGB(1, 0, 0)])
end

@testset "processors" begin
    # timm: an RGB, a grey and an RGBA image
    images = [TIMM["image/$i"] for i in 1:7]
    @test size.(images, 3) == [3, 3, 3, 3, 3, 1, 4]
    t = create_processor(joinpath(PROCESSORS, "timm--convnext_tiny.fb_in1k");
        input_size = (3, 16, 16))
    @test t(colorants.(images)) == TIMM["transform/defaults"]
    @test t(colorants(images[7])) == t(images[7])
    @test t(RGB{Float32}.(colorants(images[1]))) == t(images[1])

    # SAM 3
    numbered(prefix) = [SAM3["$prefix/$i"]
                        for i in 1:count(startswith("$prefix/"), keys(SAM3))]
    proc = create_processor(joinpath(PROCESSORS, "unsloth--sam3.1");
        size = (; height = 16, width = 24))
    ip = proc.image_processor
    for name in ("batch", "grey", "rgba")
        images = numbered("$name/image")
        out = ip(colorants.(images))
        @test out == ip(images)
        @test out.pixel_values == SAM3["$name/pixel_values"]
    end
    image = numbered("batch/image")[1]
    @test proc(; images = colorants(image)) == proc(; images = image)
    floats = RGB{Float32}.(colorants(image))
    @test ip(floats) == ip(Float32.(reinterpret(N0f8, image)))
end
