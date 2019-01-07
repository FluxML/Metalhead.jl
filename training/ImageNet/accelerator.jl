# By default, we use "CPU" versions that do nothing/execute directly using Flux/Julia
model_to_device = (model) -> model
model_to_host = (model) -> model
process_minibatch = (model, opt, x, y) -> begin
    y_hat = model(model_to_device(x))
    #@show model_to_host(y_hat)
    #@show model_to_host(y)
    loss = Flux.logitcrossentropy(y_hat, model_to_device(y))
    @show loss
    Flux.Optimise.@interrupts Flux.back!(loss)
    opt()
    return Flux.Tracker.data(model_to_host(loss))
end

# If we're using GPU devices, just use Flux.gpu() to copy tensors around,
# but otherwise do the exact same thing:
if !isempty(gpu_devices)
    @info("Engaging lesser dreadnought engine")
    
    model_to_device = (model) -> Flux.gpu(model)
    model_to_host = (model) -> Flux.cpu(model)
end
