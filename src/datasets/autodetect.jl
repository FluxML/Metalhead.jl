function datasets()
    dataset_folder = joinpath(@__DIR__, "..", "..", "datasets")
    datasets = Any[]
    for entry in readdir(dataset_folder)
        path = joinpath(dataset_folder, entry)
        isdir(path) || continue
        if entry == "ILSVRC"
            push!(datasets, ImageNet.RawFS(path, :kaggle))
            continue
        elseif entry == "cifar-10-batches-bin"
            push!(datasets, CIFAR10.BinPackedFS(path))
            continue
        end
    end
    datasets
end

"""
Get the appropriate dataset from anywhere we can find.
Available options: ImageNet
"""
function dataset(which)
    if which === ImageNet
        sets = collect(Iterators.filter(x->isa(x, ImageNet.DataSet), datasets()))
        isempty(sets) && error("No ImageNet dataset available. "*
            "See datasets/README.md for download instructions")
        return first(sets)
    elseif which == CIFAR10
        sets = collect(Iterators.filter(x->isa(x, CIFAR10.DataSet), datasets()))
        if isempty(sets)
            download(CIFAR10)
            sets = collect(Iterators.filter(x->isa(x, CIFAR10.DataSet), datasets()))
        end
        return first(sets)
    else
        error("Autodetection not supported for $(which)")
    end
end
valimgs(m::Module) = valimgs(dataset(m))
testimgs(m::Module) = testimgs(dataset(m))
trainimgs(m::Module) = trainimgs(dataset(m))

function download(which)
    if which === ImageNet
        error("ImageNet is not automatiacally downloadable. See instructions in datasets/README.md")
    elseif which == CIFAR10
        local_path = joinpath(@__DIR__, "..", "..", "datasets", "cifar-10-binary.tar.gz")
        dir_path = joinpath(@__DIR__,"..","..","datasets")
        if(!isdir(joinpath(dir_path, "cifar-10-batches-bin")))
            if(!isfile(local_path))
                Base.download("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", local_path)
            end
            run(`tar -xzvf $local_path -C $dir_path`)
        end
    else
        error("Download not supported for $(which)")
    end
end
