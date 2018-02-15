using Metalhead
using Base.Test

vgg19 = VGG19()

testx = rand(Float32, 224, 224, 3, 1)

testy = vgg19(testx)

@test testy isa AbstractArray
@test length(testy) == 1000
