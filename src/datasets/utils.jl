abstract type DataSet end
abstract type ObjectClass; end
abstract type ObjectCoords; end

Base.print(io::IO, oc::ObjectClass) = print(io, labels(typeof(oc))[oc.class])

# For classification
struct ValidationImage
    set::Type{<:DataSet}
    idx::Int
    img
    ground_truth::ObjectClass
end

struct ValData{T<:DataSet} <: AbstractVector{ValidationImage}
    set::T
end
valimgs(set::DataSet) = ValData(set)

struct TestData{T<:DataSet} <: AbstractVector{Any}
    set::T
end
testimgs(set::DataSet) = TestData(set)

struct TrainingImage
    set::Type{<:DataSet}
    idx::Int
    img
    ground_truth::ObjectClass
end

struct TrainData{T<:DataSet} <: AbstractVector{TrainingImage}
    set::T
end
trainimgs(set::DataSet) = TrainData(set)

# For localization and classification
struct ValidationBoxImage
    set::Type{<:DataSet}
    idx::Int
    img
    objects::AbstractVector{ObjectClass}
    coords::AbstractVector{ObjectCoords}
end

struct ValBoxData{T<:DataSet} <: AbstractVector{ValidationBoxImage}
    set::T
end
valimgs_box(set::DataSet) = ValBoxData(set)

struct TestBoxData{T<:DataSet} <: AbstractVector{Any}
    set::T
end
testimgs_box(set::DataSet) = TestBoxData(set)

struct TrainingBoxImage
    set::Type{<:DataSet}
    idx::Int
    img
    objects::AbstractVector{ObjectClass}
    coords::AbstractVector{ObjectCoords}
end

struct TrainBoxData{T<:DataSet} <: AbstractVector{TrainingBoxImage}
    set::T
end
trainimgs_box(set::DataSet) = TrainBoxData(set)