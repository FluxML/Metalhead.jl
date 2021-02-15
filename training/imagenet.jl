struct ImageNet
  root::String
  class_map::Dict{String, Int}
  order::Vector{String}
end

const IMG_MEAN = (0.485, 0.456, 0.406)
const IMG_STDDEV = (0.229, 0.224, 0.225)
const CMYK_IMAGES = [
  # "n01739381_1309.JPEG",
  # "n02077923_14822.JPEG",
  # "n02447366_23489.JPEG",
  # "n02492035_15739.JPEG",
  # "n02747177_10752.JPEG",
  # "n03018349_4028.JPEG",
  # "n03062245_4620.JPEG",
  # "n03347037_9675.JPEG",
  # "n03467068_12171.JPEG",
  # "n03529860_11437.JPEG",
  # "n03544143_17228.JPEG",
  # "n03633091_5218.JPEG",
  # "n03710637_5125.JPEG",
  # "n03961711_5286.JPEG",
  # "n04033995_2932.JPEG",
  # "n04258138_17003.JPEG",
  # "n04264628_27969.JPEG",
  # "n04336792_7448.JPEG",
  # "n04371774_5854.JPEG",
  # "n04596742_4225.JPEG",
  # "n07583066_647.JPEG",
  # "n13037406_4650.JPEG",
  # "ILSVRC2012_val_00019877.JPEG"
]

function ImageNet(; folder, metadata)
  class_map = Dict{String, Int}()
  order = String[]
  for line in eachline(metadata)
    s = split(line, " ")
    class_map[s[1]] = parse(Int, s[2])
    push!(order, s[1])
  end
  
  ImageNet(folder, class_map, order)
end

LearnBase.nobs(dataset::ImageNet) = length(dataset.order)
function LearnBase.getobs(dataset::ImageNet, i)
  subpath = dataset.order[i]
  file = joinpath(dataset.root, subpath)
  if basename(file) ∈ CMYK_IMAGES
    dir = dirname(file)
    run(`convert $file -colorspace RGB $(joinpath(dir, "tmp.JPEG"))`)
    data = Image(RGB.(load(joinpath(dir, "tmp.JPEG"))))
    run(`rm $(joinpath(dir, "tmp.JPEG"))`)
  else
    data = Image(RGB.(load(file)))
  end
  tfm = CenterResizeCrop((224, 224)) |> ImageToTensor() |> Normalize(IMG_MEAN, IMG_STDDEV)
  data = itemdata(apply(tfm, data))

  label = convert(Array{Float32}, Flux.onehot(dataset.class_map[subpath], labels(dataset)))

  return data, label
end
labels(dataset::ImageNet) = 1:length(unique(values(dataset.class_map)))