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
        isempty(sets) && error("No ImageNet dataset available. "*
            "See datasets/README.md for download instructions")
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
        dir_path = joinpath(@__DIR__,"..","..","datasets","cifar-10-batches-bin")
        if(!isdir(dir_path))
            Base.download("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", local_path)
            run(`tar xzf $local_path $dir_path`)
        end
        files = ["data_batch_1.bin","data_batch_2.bin","data_batch_3.bin","data_batch_4.bin","data_batch_5.bin"]
        cd(dir_path) do
            open("train_data.bin","w") do f1
                for i in files
                    if(isfile(i))
                        open(i,"r") do f2
                            write(f1, f2)
                        end
                        run(`rm ./$i`)
                    end
                end
            end
        end
    else
        error("Download not supported for $(which)")
    end
end
