struct ImageNet
  root::String
  class_map::Dict{String, Tuple{Int, String}}
  folder_order::Vector{String}
  data_map::Dict{String, Int}
end

const CMYK_IMAGES = [
  "n01739381_1309.JPEG",
  "n02077923_14822.JPEG",
  "n02447366_23489.JPEG",
  "n02492035_15739.JPEG",
  "n02747177_10752.JPEG",
  "n03018349_4028.JPEG",
  "n03062245_4620.JPEG",
  "n03347037_9675.JPEG",
  "n03467068_12171.JPEG",
  "n03529860_11437.JPEG",
  "n03544143_17228.JPEG",
  "n03633091_5218.JPEG",
  "n03710637_5125.JPEG",
  "n03961711_5286.JPEG",
  "n04033995_2932.JPEG",
  "n04258138_17003.JPEG",
  "n04264628_27969.JPEG",
  "n04336792_7448.JPEG",
  "n04371774_5854.JPEG",
  "n04596742_4225.JPEG",
  "n07583066_647.JPEG",
  "n13037406_4650.JPEG",
  "ILSVRC2012_val_00019877.JPEG"
]

function ImageNet(; folder, train=true, class_metadata=joinpath(folder, "meta.mat"))
  matdata = matread(class_metadata)["synsets"]
  class_map = Dict([wnid => (Int(class_index), class_label)
                    for (wnid, class_index, class_label, nchild) in
                      zip(matdata["WNID"],
                          matdata["ILSVRC2012_ID"],
                          matdata["words"],
                          matdata["num_children"])
                      if iszero(nchild)])

  data_dir = train ? "train" : "val"
  data_dirs = readdir(joinpath(folder, data_dir)) |> collect
  data_map = Dict{String, Int}([dir => length(readdir(joinpath(folder, "$data_dir/$dir")))
                                for dir in data_dirs])
  
  ImageNet(joinpath(folder, data_dir), class_map, data_dirs, data_map)
end

LearnBase.nobs(dataset::ImageNet) = sum(values(dataset.data_map))
function LearnBase.getobs(dataset::ImageNet, i)
  accum_images = cumsum(dataset.data_map[x] for x in dataset.folder_order)
  folder_index = findfirst(x -> x >= i, accum_images)
  isnothing(folder_index) && throw(BoundsError(dataset, i))
  wnid = dataset.folder_order[folder_index]

  ioffset = (folder_index > 1) ? i - accum_images[folder_index - 1] : i
  dir = joinpath(dataset.root, wnid)
  file = readdir(dir; join=true)[ioffset]
  if basename(file) ∈ CMYK_IMAGES
    cmd = `convert $file -colorspace RGB $(joinpath(dir, "tmp.JPEG"))`
    run(cmd)
    data = Image(RGB.(load(joinpath(dir, "tmp.JPEG"))))
    run(`rm $(joinpath(dir, "tmp.JPEG"))`)
  else
    data = Image(RGB.(load(file)))
  end
  tfm = CenterResizeCrop((224, 224)) |> ImageToTensor()
  data = itemdata(apply(tfm, data))

  label = convert(Array{Float32}, Flux.onehot(dataset.class_map[wnid][2], labels(dataset)))

  return data, label
end
labels(dataset::ImageNet) = getindex.(values(dataset.class_map), 2)