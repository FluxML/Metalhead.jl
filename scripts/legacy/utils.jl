using Functors

function imagenet_normalize(data)
    cmean = reshape(Float32[0.485, 0.456, 0.406],(1,1,3,1))
    cstd = reshape(Float32[0.229, 0.224, 0.225],(1,1,3,1))
    return (data .- cmean) ./ cstd
end


# Julia -> Python 

function jl2np(x::Array)
    x = permutedims(x, ndims(x):-1:1)
    x_np = Py(x).to_numpy()
    return x_np
end

jl2th(x::Array) = torch.from_numpy(jl2np(x))

# Python -> Julia

function np2jl(x::Py)
    x_jl = pyconvert(Array, x)
    x_jl = permutedims(x_jl, ndims(x_jl):-1:1)
    return x_jl
end

function th2jl(x::Py)
    x_jl = pyconvert(Array, x.detach().numpy())
    x_jl = permutedims(x_jl, ndims(x_jl):-1:1)
    return x_jl
end

py2jl(x::Py) = pyconvert(Any, x)
