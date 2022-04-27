function ASPPConv(kernel, channels::Pair{Int,Int}; rate)
   # @show channels
   Chain(
 	Conv(kernel, channels, pad = rate, dilation = rate),
 	BatchNorm(channels[2], relu))
 end

 struct ASPPPooling{T}
   chain::T
 end

 @functor ASPPPooling

 function ASPPPooling(channels::Pair)
   ASPPPooling(Chain(AdaptiveMeanPool((1,1)),
 	Conv((1,1), channels),
 	BatchNorm(channels[2], relu)))
 end

 function (asp::ASPPPooling)(x)
   out = asp.chain(x)
   upsample_bilinear(out, size = size(x)[1:2])
 end

 struct ASPP{T,S}
   par::T
   project::S
 end

 @functor ASPP

 function ASPP(in_channels::Int, atrous_rates, out_channels = 256)
   c1 = Chain(Conv((1,1), in_channels => out_channels),
 	      BatchNorm(out_channels, relu))

   ls = [ASPPConv((3,3), in_channels => out_channels; rate = rate) for rate in atrous_rates]


   par = Parallel((x...) -> cat(x..., dims = 3), c1, ls...,  ASPPPooling(in_channels => out_channels))
   project = Chain(Conv((1,1), ((length(ls)+1+1) * out_channels) => out_channels),
 		   BatchNorm(out_channels),
 		   Dropout(0.5))
   ASPP(par, project)
 end

 function (aspp::ASPP)(x)
   aspp.project(aspp.par(x))
 end

 function DeepLabV3Head(in_channels, classes)
   Chain(ASPP(in_channels, [12, 24, 36]),
 	Conv((3,3), 256 => 256, pad = 1),
 	BatchNorm(256, relu),
 	Conv((1,1), 256 => classes))
 end

 function FCNHead(in_channels, channels)
   inter_channels = Int(in_channels / 4)
   Chain(Conv((3,3), in_channels => inter_channels, pad = 1),
 	BatchNorm(inter_channels, relu),
 	Dropout(0.1),
 	Conv((1,1), inter_channels => channels))
 end

 struct DeepLabV3
   resnet::Chain
   head::Chain
   aux::Chain
 end

 @functor DeepLabV3

 function DeepLabV3(in_channels::Int, classes; backbone = ResNet().layers[1:end-4])

   # backbone = model[1][1:end-3]
   dh = DeepLabV3Head(in_channels, classes)
   aux = FCNHead(1024, 21)
   DeepLabV3(backbone, dh, aux)
 end

 function (dl::DeepLabV3)(x)
   ishape = size(x)[1:2]
   feats = dl.resnet(x)
   upsample_bilinear(dl.head(feats), size = ishape)
 end
