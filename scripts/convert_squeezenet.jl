using BSON, Flux, JLD2, Metalhead

function collect_weights!(weights, x::AbstractDict)
    if get(x, :tag, "") == "struct" && x[:type][:name] == ["Flux", "Conv"]
        for array in x[:data][2:3]
            push!(weights, BSON.raise_recursive(deepcopy(array), Metalhead))
        end
    elseif haskey(x, :data)
        collect_weights!(weights, x[:data])
    end
    return weights
end

function collect_weights!(weights, xs::AbstractVector)
    foreach(x -> collect_weights!(weights, x), xs)
    return weights
end

collect_weights!(weights, x) = weights

source, destination = ARGS
weights = collect_weights!(Array{Float32}[], BSON.parse(source)[:model])
length(weights) == 52 || error("Expected 52 SqueezeNet weight arrays")
model = SqueezeNet()
Flux.loadmodel!(model, weights)
Flux.trainables(model) == weights || error("SqueezeNet weights did not round-trip")
JLD2.jldsave(destination; model_state = weights)
