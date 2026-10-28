using Metalhead, Test
using Flux: Flux, Dense, Chain
using ChainRulesCore: unthunk

const stage_outputs = Metalhead.Timm.stage_outputs

@testset "vector stage outputs" begin
    layers = [Dense(3 => 3, tanh) for _ in 1:32]
    x = rand(Float32, 3, 2)
    @test stage_outputs(layers, x, 0) == ()
    @test stage_outputs(layers, x, 3) ≈ collect(Flux.activations(Chain(layers[1:3]...), x))
    @test stage_outputs(layers, x) isa Vector
    loss(ls, x) = sum(abs2, stage_outputs(ls, x, 3)[1]) +
                  sum(abs2, stage_outputs(ls, x, 3)[3])
    reference(ls, x) = sum(abs2, ls[1](x)) + sum(abs2, ls[3](ls[2](ls[1](x))))
    gl, gx = Flux.gradient(loss, layers, x)
    rl, rx = Flux.gradient(reference, layers, x)
    @test gx ≈ rx
    for i in 1:3
        @test unthunk(gl[i].weight) ≈ unthunk(rl[i].weight)
        @test unthunk(gl[i].bias) ≈ unthunk(rl[i].bias)
    end
    @test all(isnothing, gl[4:end])
end
