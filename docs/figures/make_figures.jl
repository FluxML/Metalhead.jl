# The figures of the documentation, written to docs/src/assets. From the repository root:
#
#     julia --project=docs/figures -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#     julia --project=docs/figures docs/figures/make_figures.jl              # all of them
#     julia --project=docs/figures docs/figures/make_figures.jl sam3 classify
#
# Don't commit the `[sources]` entry that `Pkg.develop` writes into Project.toml.
#
# The weights download to the Hugging Face cache. DINOv2, SAM 3.1 and FLUX.2 [klein] run on
# the GPU when CUDA finds one; FLUX.2 [klein] 4B needs 16 GB of it.

using Metalhead, Flux, HuggingFaceTokenizers, CUDA, cuDNN
using Metalhead.Diffusers: encode_prompt, compute_empirical_mu, set_timesteps
using CairoMakie, FileIO, ImageIO, ImageCore, ImageTransformations
using Downloads, LinearAlgebra, Printf, Random, Statistics

const ASSETS = normpath(joinpath(@__DIR__, "..", "src", "assets"))
const CACHE = joinpath(@__DIR__, "cache")

const SURFACE = colorant"#fcfcfb"
const INK = colorant"#0b0b0b"
const INK2 = colorant"#52514e"
const BLUE = colorant"#2a78d6"
const INSTANCE_COLORS = [colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a"]
const OTHER = colorant"#8a8984"
const RAMP = parse.(RGB{N0f8},
    ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
const MONO = "DejaVu Sans Mono"

const THEME = Theme(; backgroundcolor = SURFACE, textcolor = INK, fontsize = 15,
    figure_padding = 8,
    Axis = (; backgroundcolor = SURFACE, xgridvisible = false, ygridvisible = false,
        topspinevisible = false, rightspinevisible = false, leftspinecolor = INK2,
        bottomspinecolor = INK2, xtickcolor = INK2, ytickcolor = INK2,
        xticklabelcolor = INK2, yticklabelcolor = INK),
    Label = (; color = INK2))

const URLS = Dict(
    "labrador.jpg" => "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg",
    "cats.jpg" => "http://images.cocodataset.org/val2017/000000039769.jpg",
    "imagenet_classes.txt" => "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")

function cached(name)
    path = joinpath(CACHE, name)
    isfile(path) || (mkpath(CACHE); Downloads.download(URLS[name], path))
    return path
end

labrador() = load(cached("labrador.jpg"))
cats() = load(cached("cats.jpg"))

# A width × height × 3 array as an image, a height × width matrix of colours
as_image(x) = Metalhead.image_matrix(clamp.(x, 0, 1))

# The input that a timm transform makes, without its normalisation
function denormalize(transform, x)
    mean, std = (reshape(collect(Float32, v), 1, 1, 3)
                 for v in (transform.mean, transform.std))
    return as_image(x[:, :, :, 1] .* std .+ mean)
end

# The centre of an image at an aspect ratio of 4:3, resized to 800 × 600
function thumbnail(img)
    h, w = size(img)
    if 3w > 4h
        cw = 4h ÷ 3
        img = img[:, (w - cw) ÷ 2 .+ (1:cw)]
    else
        ch = 3w ÷ 4
        img = img[(h - ch) ÷ 2 .+ (1:ch), :]
    end
    return imresize(img, (600, 800))
end

function save_asset(name, img::AbstractMatrix{<:Colorant}; quality = 85)
    path = joinpath(ASSETS, name)
    FileIO.save(path, RGB{N0f8}.(img); quality)
    return path
end

# Charts as PNG, photographs as JPEG, both at twice the size of the figure
function save_asset(name, fig::Figure; px_per_unit = 2)
    endswith(name, ".png") || return save_asset(name, colorbuffer(fig; px_per_unit))
    path = joinpath(ASSETS, name)
    FileIO.save(path, fig; px_per_unit)
    return path
end

# An axis that shows `img` in its pixel coordinates, y downwards
function image_axis(pos, img; interpolate = true, kwargs...)
    h, w = size(img)
    ax = Axis(pos; aspect = DataAspect(), yreversed = true, limits = (0, w, 0, h),
        kwargs...)
    hidedecorations!(ax)
    hidespines!(ax)
    image!(ax, 0 .. w, 0 .. h, permutedims(img); interpolate)
    return ax
end

# Horizontal bars, labelled on the y axis and at their ends
function bar_axis(pos, labels, values; kwargs...)
    n = length(values)
    ax = Axis(pos; yticks = (1:n, labels), yreversed = true, yticksvisible = false,
        limits = (0, 1.2 * maximum(values), 0.4, n + 0.6), kwargs...)
    hidexdecorations!(ax)
    hidespines!(ax, :t, :r, :b)
    barplot!(ax, 1:n, values; direction = :x, color = BLUE, gap = 0.6)
    text!(ax, values, 1:n; text = [@sprintf("%.1f%%", 100v) for v in values],
        align = (:left, :center), offset = (6, 0), color = INK, fontsize = 15)
    return ax
end

function top_classes(repo, img, labels; k = 5)
    model = create_model(repo)
    transform = create_processor(repo)
    p = vec(softmax(model(transform(img))))
    top = sortperm(p; rev = true)[1:k]
    return labels[top], p[top]
end

function classify()
    img = labrador()
    labels = readlines(cached("imagenet_classes.txt"))
    names, probs = top_classes("timm/resnet18.a1_in1k", img, labels)
    foreach((n, p) -> @printf("  resnet18.a1_in1k  %-26s %.4f\n", n, p), names, probs)
    foreach((n, p) -> @printf("  convnext_tiny     %-26s %.4f\n", n, p),
        top_classes("timm/convnext_tiny.fb_in1k", img, labels)...)

    fig = Figure()
    h, w = size(img)
    image_axis(fig[1, 1], img; width = 440, height = 440h / w)
    bar_axis(fig[1, 2], names, probs; width = 330, height = 300)
    colgap!(fig.layout, 24)
    resize_to_layout!(fig)
    save_asset("classify.jpg", fig)

    thumb = Figure(; size = (800, 600), figure_padding = 0)
    ax = image_axis(thumb[1, 1], thumbnail(img))
    chip!(ax, 24, 64, @sprintf("%s %.1f%%", names[1], 100 * probs[1]), BLUE; scale = 1,
        xmax = 800, fontsize = 26, pad = 7)
    save_asset("showcase-classify.jpg", thumb; px_per_unit = 1)
    return nothing
end

function transform_figure()
    transform = create_processor("timm/resnet18.a1_in1k")
    img = labrador()
    x = transform(img)
    # where the centre crop of timm's eval transform falls in the photo
    h, w = size(img)
    _, crop_h, crop_w = transform.input_size
    short = floor(Int, crop_h / transform.crop_pct)
    rw, rh = w <= h ? (short, trunc(Int, short * h / w)) :
             (trunc(Int, short * w / h), short)
    left, top = round(Int, (rw - crop_w) / 2), round(Int, (rh - crop_h) / 2)
    x1, x2 = (left, left + crop_w) .* (w / rw)
    y1, y2 = (top, top + crop_h) .* (h / rh)

    fig = Figure()
    ax = image_axis(fig[1, 1], img; width = 470, height = 470h / w)
    outside = [Rect2f(0, 0, w, y1), Rect2f(0, y2, w, h - y2), Rect2f(0, y1, x1, y2 - y1),
        Rect2f(x2, y1, w - x2, y2 - y1)]
    poly!(ax, outside; color = (SURFACE, 0.6))
    lines!(ax, [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1]; color = BLUE, linewidth = 2)
    Label(fig[2, 1],
        "$w × $h  →  shorter side $short ($rw × $rh)  →  " *
        "centre crop $crop_w × $crop_h")
    image_axis(fig[1, 2], denormalize(transform, x); interpolate = false,
        width = 470h / w, height = 470h / w)
    Label(fig[2, 2], join(size(x), " × "))
    colgap!(fig.layout, 24)
    rowgap!(fig.layout, 8)
    resize_to_layout!(fig)
    save_asset("transform.jpg", fig)
    return nothing
end

function features()
    repo = "timm/resnet18.a1_in1k"
    model = create_model(repo)
    transform = create_processor(repo)
    x = transform(labrador())
    info = feature_info(model)[2:5]
    maps = forward_intermediates(model, x; indices = 2:5, intermediates_only = true)

    side = 170
    fig = Figure()
    image_axis(fig[1, 1], denormalize(transform, x); width = side, height = side)
    Label(fig[2, 1], "input\n" * join(size(x)[1:3], " × "))
    for (k, (f, y)) in enumerate(zip(info, maps))
        # the norm of the features at each position, each map on its own scale
        norms = sqrt.(dropdims(sum(abs2, y[:, :, :, 1]; dims = 3); dims = 3))
        ax = Axis(fig[1, k + 1]; aspect = DataAspect(), yreversed = true, width = side,
            height = side)
        hidedecorations!(ax)
        hidespines!(ax)
        heatmap!(ax, norms; colormap = RAMP)
        Label(fig[2, k + 1], "stride $(f.reduction)\n" * join(size(y)[1:3], " × "))
        @printf("  %s: stride %d, %s\n", f.module_name, f.reduction, size(y))
    end
    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 8)
    resize_to_layout!(fig)
    save_asset("features.png", fig)
    return nothing
end

# The projections of the rows of X on their first k principal axes, with signs that don't
# depend on the LAPACK
function principal_components(X, k)
    X = X .- mean(X; dims = 1)
    V = svd(X).V[:, 1:k]
    V .*= sign.(sum(V .^ 3; dims = 1))
    return X * V
end

rescale(v) = (v .- minimum(v)) ./ (maximum(v) - minimum(v))

function dinov2_pca()
    repo = "timm/vit_small_patch14_reg4_dinov2.lvd142m"
    device = gpu_device()
    model = device(create_model(repo; dynamic_img_size = true))
    img = cats()
    transform = create_processor(repo; input_size = (3, 672, 896))
    x = transform(img)
    y = cpu(only(forward_intermediates(model, device(x); indices = 1, norm = true,
        intermediates_only = true)))
    w, h, c = size(y)
    @printf("  input %s, tokens %d × %d × %d\n", size(x), w, h, c)
    X = Float64.(reshape(y, w * h, c))

    # the first component separates the objects from the background, here the border
    pc1 = vec(principal_components(X, 1))
    border = vec([i in (1, w) || j in (1, h) for i in 1:w, j in 1:h])
    mean(pc1[border]) > 0 && (pc1 .= -pc1)
    foreground = pc1 .> 0
    pcs = principal_components(X[foreground, :], 3)
    rgb = zeros(RGB{Float32}, w * h)
    rgb[foreground] = RGB{Float32}.(rescale.(eachcol(pcs))...)
    pca = permutedims(reshape(rgb, w, h))

    fig = Figure()
    width, height = 450, 450 * size(img, 1) / size(img, 2)
    image_axis(fig[1, 1], img; width, height)
    Label(fig[2, 1], "input, resized to $(size(x, 1)) × $(size(x, 2))")
    image_axis(fig[1, 2], pca; interpolate = false, width, height)
    Label(fig[2, 2], "$w × $h patch tokens, 3 principal components as RGB")
    colgap!(fig.layout, 16)
    rowgap!(fig.layout, 8)
    resize_to_layout!(fig)
    save_asset("dinov2_pca.jpg", fig)
    save_asset("showcase-features.jpg", thumbnail(repeat(pca; inner = (25, 25))))
    return nothing
end

# A label chip, white text on `color`, above the top left corner (x, y) of a box, or below
# it at the top of the image, in the data units of `ax`, which show `scale` pixels each
function chip!(ax, x, y, str, color; scale, xmax, fontsize = 14, pad = 4)
    tw = Makie.text_bb(str, Makie.defaultfont(), fontsize).widths[1]
    cw, ch = (tw + 2pad) / scale, (fontsize + 2pad) / scale
    x = min(x, xmax - cw)
    y = y >= ch ? y - ch : y
    poly!(ax, Rect2f(x, y, cw, ch); color)
    text!(ax, x + pad / scale, y + ch / 2; text = str, color = :white, fontsize,
        align = (:left, :center))
    return nothing
end

function draw_objects!(ax, objects, name; scale, fontsize = 14, linewidth = 2)
    w, h = size(objects.masks)
    colors = [i <= length(INSTANCE_COLORS) ? INSTANCE_COLORS[i] : OTHER
              for i in eachindex(objects.scores)]
    for (i, color) in reverse(collect(enumerate(colors)))
        mask = objects.masks[:, :, i]
        image!(ax, 0 .. w, 0 .. h, map(m -> RGBAf(color, m ? 0.45 : 0), mask);
            interpolate = false)
        contour!(ax, 0.5:(w - 0.5), 0.5:(h - 0.5), Float32.(mask); levels = [0.5],
            color, linewidth)
    end
    for (i, color) in enumerate(colors)
        label = strip(@sprintf("%s %.2f", name, objects.scores[i]))
        chip!(ax, objects.boxes[1, i], objects.boxes[2, i], label, color; scale,
            xmax = w, fontsize)
    end
    return nothing
end

function sam3()
    device = gpu_device()
    repo = "unsloth/sam3.1"
    model = device(create_model(repo))
    proc = create_processor(repo)
    img = cats()
    box = (340, 20, 638, 376)
    prompts = [
        ("cat", (; text = "a cat"), "text = \"a cat\""),
        ("remote", (; text = "a remote control"), "text = \"a remote control\""),
        ("", (; input_boxes = [[box]]), "input_boxes = [[$box]]")]
    width = 300
    fig = Figure()
    for (k, (name, prompt, caption)) in enumerate(prompts)
        inputs = proc(; images = img, prompt...)
        objects = only(post_process_instance_segmentation(proc, model(device(inputs));
            threshold = 0.5, target_sizes = inputs.original_sizes))
        println("  ", caption)
        for i in eachindex(objects.scores)
            @printf("    score %.3f  box (%.1f, %.1f, %.1f, %.1f)\n", objects.scores[i],
                objects.boxes[:, i]...)
        end
        ax = image_axis(fig[1, k], img; width, height = width * 3 / 4)
        draw_objects!(ax, objects, name; scale = width / size(img, 2))
        if haskey(prompt, :input_boxes)
            x1, y1, x2, y2 = box
            lines!(ax, [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1]; color = INK,
                linewidth = 2, linestyle = :dash)
        end
        Label(fig[2, k], caption; font = MONO, fontsize = 14)
        if k == 1
            thumb = Figure(; size = (800, 600), figure_padding = 0)
            draw_objects!(image_axis(thumb[1, 1], img), objects, name;
                scale = 800 / size(img, 2), fontsize = 22, linewidth = 3)
            save_asset("showcase-segment.jpg", thumb; px_per_unit = 1)
        end
    end
    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 8)
    resize_to_layout!(fig)
    save_asset("sam3.jpg", fig)
    return nothing
end

function next_token()
    repo = "HuggingFaceTB/SmolLM2-135M"
    model = create_model(repo)
    tok = create_processor(repo)
    p = softmax(model(tok("The capital of France is"))[:, end])
    top = sortperm(p; rev = true)[1:10]
    tokens = [repr(decode(tok, i)) for i in top]
    foreach((t, v) -> @printf("  %-14s %.4f\n", t, v), tokens, p[top])
    fig = Figure()
    bar_axis(fig[1, 1], tokens, p[top]; width = 520, height = 380,
        yticklabelfont = MONO)
    resize_to_layout!(fig)
    save_asset("next_token.png", fig)
    return nothing
end

const FLUX2_PROMPTS = [
    "lighthouse" => "A watercolor painting of a lighthouse on a rocky cliff at sunset",
    "spiderweb" => "A macro photograph of dew drops on a spider web at dawn, " *
                   "shallow depth of field",
    "library" => "An isometric illustration of a tiny cozy library with warm lights",
    "ramen" => "A bowl of ramen on a wooden table, studio food photography"]
const FLUX2_SHOWCASE = "spiderweb"

const FLUX2 = Ref{Any}(nothing)

# The photograph that the vision-language tutorial asks about
vision_language() = (save_asset("cats.jpg", cats()); nothing)

function flux2_pipeline()
    if FLUX2[] === nothing
        FLUX2[] = gpu(create_model("black-forest-labs/FLUX.2-klein-4B"; eltype = BFloat16))
    end
    return FLUX2[]
end

# The seed of each image is its index in FLUX2_PROMPTS
function flux2_gallery()
    pipe = flux2_pipeline()
    images = map(enumerate(FLUX2_PROMPTS)) do (seed, (name, prompt))
        t = @elapsed image = pipe(prompt; width = 1024, height = 768,
            num_inference_steps = 4, rng = Xoshiro(seed))
        @printf("  %-10s seed %d, %.1f s, %s: %s\n", name, seed, t, size(image), prompt)
        img = as_image(cpu(image)[:, :, :, 1])
        name == FLUX2_SHOWCASE &&
            save_asset("showcase-generate.jpg", imresize(img, (600, 800)))
        return img
    end
    w, gap = 796, 8
    h = 3w ÷ 4
    grid = fill(RGB{N0f8}(SURFACE), 2h + gap, 2w + gap)
    for (k, img) in enumerate(images)
        i, j = fldmod1(k, 2)
        grid[(i - 1) * (h + gap) .+ (1:h), (j - 1) * (w + gap) .+ (1:w)] = imresize(img,
            (h, w))
    end
    save_asset("flux2_gallery.jpg", grid)
    return nothing
end

# The clean image that the pipeline predicts at each step, `latents - sigma * flow`. That of
# step k is the image of the first k steps of the schedule followed by a step to noise
# level 0. With k steps, the pipeline shifts the noise levels by another `mu`, which the
# levels it is given undo.
function flux2_steps()
    pipe = flux2_pipeline()
    seed = 1
    prompt = last(FLUX2_PROMPTS[seed])
    width, height, steps = 1024, 768, 4
    embeds = encode_prompt(pipe, prompt)
    seq_len = (width ÷ 16) * (height ÷ 16)
    _, sigmas = set_timesteps(pipe.scheduler, steps;
        sigmas = range(1, 1 / steps; length = steps),
        mu = compute_empirical_mu(seq_len, steps))
    fig = Figure()
    for k in 1:steps
        m = exp(compute_empirical_mu(seq_len, k))
        levels = @. 1 / (1 + m * (1 / sigmas[1:k] - 1))
        image = pipe(embeds; width, height, num_inference_steps = k, sigmas = levels,
            rng = Xoshiro(seed))
        image_axis(fig[1, k], as_image(cpu(image)[:, :, :, 1]); width = 236,
            height = 177)
        Label(fig[2, k], @sprintf("step %d · σ = %.2f", k, sigmas[k]))
    end
    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 6)
    resize_to_layout!(fig)
    save_asset("flux2_steps.jpg", fig)
    return nothing
end

const FIGURES = [
    "classify" => classify,
    "transform" => transform_figure,
    "features" => features,
    "dinov2_pca" => dinov2_pca,
    "sam3" => sam3,
    "next_token" => next_token,
    "vision_language" => vision_language,
    "flux2_gallery" => flux2_gallery,
    "flux2_steps" => flux2_steps]

function main(names = ARGS)
    CairoMakie.activate!()
    set_theme!(THEME)
    mkpath(ASSETS)
    for name in (isempty(names) ? first.(FIGURES) : names)
        figure = Dict(FIGURES)[name]
        println(name)
        figure()
        GC.gc()
        CUDA.functional() && CUDA.reclaim()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
