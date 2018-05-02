module CIFAR10

import ..Metalhead
import ..Metalhead: testimgs
import ..Metalhead: ValidationImage, TrainingImage, ValData, TestData, TrainData, ObjectClass, labels

abstract type DataSet <: Metalhead.DataSet end
using Images
using ColorTypes
using ColorTypes: N0f8

const C10Labels = ["airplane", "automobile", "bird", "cat", "deer", "dog",
    "frog", "horse", "ship", "truck"]

struct C10Class <: ObjectClass
    class::Int
end
labels(::Type{C10Class}) = C10Labels

struct BinPackedFS <: DataSet
    folder::String
end

Base.size(v::ValData{<:DataSet}) = (10000,)

Base.size(v::TrainData{<:DataSet}) = (50000,)

testimgs(::DataSet) = error("CIFAR10 does not specify a test set (test_batch is considered the validation set)")

function bytes_to_image(bytes::Vector{UInt8})
    channelwise = reshape(bytes, (1024, 3))
    transpose(reshape(
        RGB.((reinterpret.(N0f8, channelwise[:, i]) for i = 1:3)...),
        (32, 32)))
end

function Base.getindex(v::ValData{BinPackedFS}, i::Integer)
    ValidationImage(DataSet, i, open(joinpath(v.set.folder, "test_batch.bin")) do f
        seek(f, (i-1)*3073)
        label = read(f, UInt8)
        bytes = read(f, 3072)
        bytes_to_image(bytes), C10Class(label+1)
    end...)
end

function Base.getindex(v::TrainData{BinPackedFS}, i::Integer)
    batch, num = divrem(i-1,10000)
    file = "data_batch_$(batch+1).bin"
    TrainingImage(DataSet, i, open(joinpath(v.set.folder, file)) do f
        seek(f, (num)*3073)
        label = read(f, UInt8)
        bytes = read(f, 3072)
        bytes_to_image(bytes), C10Class(label+1)
    end...)
end

end
