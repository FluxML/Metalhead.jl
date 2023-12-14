function loss_and_accuracy(data_loader, model, device; limit = nothing)
    acc = 0
    ls = 0.0f0
    num = 0
    i = 0
    for (x, y) in data_loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        ls += logitcrossentropy(ŷ, y, agg=sum)
        acc += sum(onecold(ŷ) .== onecold(y))
        num +=  size(x)[end]
        if limit !== nothing
            i == limit && break
            i += 1
        end
    end
    return ls / num, acc / num
end

function load_cifar10(; batchsize=1000)
    @info "loading CIFAR-10 dataset"
    train_dataset, test_dataset = CIFAR10(split=:train), CIFAR10(split=:test)
    train_x, train_y = train_dataset[:]
    test_x, test_y = test_dataset[:]
    @assert train_dataset.metadata["class_names"] == test_dataset.metadata["class_names"]
    labels = train_dataset.metadata["class_names"]

    # CIFAR10 label indices seem to be zero-indexed
    train_y .+= 1
    test_y .+= 1

    train_y_ohb = Flux.onehotbatch(train_y, eachindex(labels))
    test_y_ohb = Flux.onehotbatch(test_y, eachindex(labels))

    train_loader = Flux.DataLoader((data=train_x, labels=train_y_ohb); batchsize, shuffle=true)
    test_loader = Flux.DataLoader((data=test_x, labels=test_y_ohb); batchsize)

    return train_loader, test_loader, labels
end

function _train(model, train_loader, test_loader; epochs = 45, device = gpu, limit=nothing, gpu_gc=true, gpu_stats=false, show_plots=false, to=TimerOutput())

    model = model |> device

    opt = Optimisers.Adam()
    state = Optimisers.setup(opt, model)

    train_loss_hist, train_acc_hist = Float64[], Float64[]
    test_loss_hist, test_acc_hist = Float64[], Float64[]

    @info "starting training"
    for epoch in 1:epochs
        i = 0
        @showprogress "training epoch $epoch/$epochs" for (x, y) in train_loader
            x, y = x |> device, y |> device
            @timeit to "batch step" begin
                gs, _ = gradient(model, x) do m, _x
                    logitcrossentropy(m(_x), y)
                end
                state, model = Optimisers.update(state, model, gs)
            end

            device === gpu && gpu_stats && CUDA.memory_status()
            if limit !== nothing
                i == limit && break
                i += 1
            end
        end

        @info "epoch $epoch complete. Testing..."
        train_loss, train_acc = loss_and_accuracy(train_loader, model, device; limit)
        @timeit to "testing" test_loss, test_acc = loss_and_accuracy(test_loader, model, device; limit)
        @info map(x->round(x, digits=3), (; train_loss, train_acc, test_loss, test_acc))

        if show_plots
            push!(train_loss_hist, train_loss); push!(train_acc_hist, train_acc);
            push!(test_loss_hist, test_loss); push!(test_acc_hist, test_acc);
            plt = lineplot(1:epoch, train_loss_hist, name = "train_loss", xlabel="epoch", ylabel="loss")
            lineplot!(plt, 1:epoch, test_loss_hist, name = "test_loss")
            display(plt)
            plt = lineplot(1:epoch, train_acc_hist, name = "train_acc", xlabel="epoch", ylabel="acc")
            lineplot!(plt, 1:epoch, test_acc_hist, name = "test_acc")
            display(plt)
        end
        if device === gpu && gpu_gc
            GC.gc() # GPU will OOM without this
        end
    end
end

# because Flux stacktraces are ludicrously big on <1.10 so don't show them
function train(args...;kwargs...)
    try
        _train(args...; kwargs...)
    catch ex
        # rethrow()
        println()
        @error sprint(showerror, ex)
        GC.gc()
        return false
    end
end
