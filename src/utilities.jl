function conv_bn(kernelsize::Tuple{Int64,Int64}, inplanes::Int64, outplanes::Int64;
                 stride::Int64=1, pad::Union{Int64,Tuple{Int64,Int64}}=0,
                 usebias::Bool=true, rev::Bool=false)
  conv_layer = []
  if usebias
    push!(conv_layer, Conv(kernelsize, inplanes => outplanes, stride=stride, pad=pad))
  else
    push!(conv_layer, Conv(kernelsize, inplanes => outplanes, stride=stride, pad=pad, bias=Flux.Zeros()))
  end

  if rev
    push!(conv_layer, BatchNorm(inplanes, relu))
    return reverse(Tuple(conv_layer))
  end

  push!(conv_layer, BatchNorm(outplanes, relu))
  return Tuple(conv_layer)
end

cat_channels(x, y) = cat(x, y; dims=3)

weights(model) = BSON.load(joinpath(@artifact_str(model), "$model.bson"), @__MODULE__)[:weights]
pretrain_error(model) = throw(ArgumentError("No pre-trained weights available for $model."))