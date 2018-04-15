module ImageNet

import ..Metalhead
import ..Metalhead: ValidationImage, ValData, TestData, TrainData, ObjectClass

using Images

import ..valimgs, ..testimgs, ..labels

# Read ImageNet Metadata
const imagenet_val_labels_file = joinpath(@__DIR__, "..", "..", "datasets", "meta", "ILSVRC_val_labels.txt")
const imagenet_val_labels = String[strip(line) for line in eachline(imagenet_val_labels_file)]
const imagenet_labels = String[]
const synset_mapping = Dict{String, Int}()

const synset_mapping_file = joinpath(@__DIR__, "..", "..", "datasets", "meta", "ILSVRC_synset_mappings.txt")
for (idx, line) in enumerate(eachline(synset_mapping_file))
    synset = line[1:9]
    label = line[11:end]
    push!(imagenet_labels, label)
    synset_mapping[synset] = idx
end

struct ImageNet1k <: ObjectClass
    class::Int
end
labels(::Type{ImageNet1k}) = imagenet_labels

function Base.print(io::IO, class::ImageNet1k)
    print(io, imagenet_labels[class.class])
end

abstract type DataSet <: Metalhead.DataSet end

Base.size(v::ValData{<:DataSet}) = (50000,)
Base.size(v::TestData{<:DataSet}) = (100000,)

struct RawFS <: DataSet
    train_folder::String
    test_folder::String
    val_folder::String
end
function Base.show(io::IO, set::RawFS)
    println(io, "ImageNet DataSet on FileSystem")
    println(io, "      Train: $(set.train_folder)")
    println(io, "       Test: $(set.test_folder)")
    println(io, " Validation: $(set.val_folder)")
end

function RawFS(folder::String, layout=:none)
    if layout == :kaggle
        base = joinpath(folder, "Data", "CLS-LOC")
        return RawFS(
            joinpath(base, "train"),
            joinpath(base, "test"),
            joinpath(base, "val"),
        )
    else
        error("Unrecognized fs layout $(layout)")
    end
end

function Base.getindex(val::ValData{RawFS}, i::Int)
    fname = joinpath(val.set.val_folder, "ILSVRC2012_val_$(lpad(string(i), 8, '0')).JPEG")
    label = synset_mapping[imagenet_val_labels[i]][1]
    ValidationImage(DataSet, i, fname, ImageNet1k(label))
end
Base.getindex(val::TestData{RawFS}, i::Int) = joinpath(val.set.test_folder, "ILSVRC2012_test_$(lpad(string(i), 8, '0')).JPEG")

end
