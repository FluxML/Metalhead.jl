abstract type DataSet end
abstract type ObjectClass; end
Base.print(io::IO, oc::ObjectClass) = print(io, labels(typeof(oc))[oc.class])

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

struct TrainData{T<:DataSet} <: AbstractVector{Any}
    set::T
end
