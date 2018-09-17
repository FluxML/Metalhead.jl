abstract type ClassificationModel{Class} end
labels(model::Type{<:ClassificationModel{Class}}) where {Class} = labels(Class)
labels(model::ClassificationModel{Class}) where {Class} = labels(Class)
Base.broadcastable(m::ClassificationModel) = Ref(m)
