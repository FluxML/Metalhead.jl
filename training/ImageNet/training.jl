using Metalhead, Flux, BSON, CUDAnative
using Statistics, Printf

### Stuff that Flux should probably have
"""
    save_model(model, filename::AbstractString)

Iterates over all params() of a model, writing them out to a BSON file saved at
the given `filename` path.
"""
function save_model(model, filename::AbstractString)
    model_state = Dict(
        :weights => Flux.Tracker.data.(params(model))
    )
    open(filename, "w") do io
        BSON.bson(io, model_state)
    end
end

"""
    load_model!(model, filename::AbstractString)

Loads parameters from the given BSON weights file, shoves them into the given
`model` using `Flux.loadparams!()`.
"""
function load_model!(model, filename::AbstractString)
    weights = BSON.load(filename)[:weights]
    Flux.loadparams!(model, weights)
    return model
end
###


"""
    TrainState

Structure that contains all information and state related to a training run,
such that training can continue if this struct is loaded from disk.
"""
mutable struct TrainState
    # Things setup at the beginning, that shouldn't really change
    model_name
    model_args
    model_kwargs
    model

    opt_name
    opt_args
    opt_kwargs
    opt
    
    train_dataset
    val_dataset
    max_epochs
    patience

    # Things that change regularly.  Note that train_loss_history is by minibatch,
    # whereas val_loss_history is by epoch.
    train_loss_history
    val_loss_history
    epoch
end

function TrainState(model_name, model_args, model_kwargs, model,
                    opt_name, opt_args, opt_kwargs, opt,
                    train_dataset, val_dataset,
                    max_epochs, patience)
    return TrainState(
        # Model stuff 
        model_name,
        model_args,
        model_kwargs,
        model,

        # Optimiser stuff
        opt_name,
        opt_args,
        opt_kwargs,
        opt,

        # Dataset stuff
        train_dataset,
        val_dataset,

        # How many epochs we will run total
        max_epochs,
        patience,

        # Dynamic stuff
        Vector{Float64}[],
        Float64[],
        1,
    )
end

function save_train_state(ts::TrainState, output_dir::AbstractString,
                          should_save_weights::Bool = true)
    if should_save_weights
        save_model(cpu(ts.model), joinpath(output_dir, "weights.bson"))
    end
    # Next, save training info
    train_status = Dict(
        # Model weights are saved separately
        :model_name => ts.model_name,
        :model_args => ts.model_args,
        :model_kwargs => ts.model_kwargs,
        # Can't save the optimizer yet, we'll have to add that later.
        #:opt => cpu(ts.opt),
        :opt_name => ts.opt_name,
        :opt_args => ts.opt_args,
        :opt_kwargs => ts.opt_kwargs,

        # Save our dataset such that we can recreate it
        :train_dataset_type => typeof(ts.train_dataset),
        :train_dataset_args => freeze_args(ts.train_dataset),
        :val_dataset_type => typeof(ts.val_dataset),
        :val_dataset_args => freeze_args(ts.val_dataset),

        # Save various statistics
        :max_epochs => ts.max_epochs,
        :train_loss_history => ts.train_loss_history,
        :val_loss_history => ts.val_loss_history,
        :epoch => ts.epoch,
        :patience => ts.patience,
    )
    open(joinpath(output_dir, "status.bson"), "w") do io
        BSON.bson(io, train_status)
    end
end

"""
    load_train_state(output_dir::AbstractString)

Given a previous training run, recreate a `TrainState` object from the saved
BSON files within the `output_dir`.
"""
function load_train_state(output_dir::AbstractString)
    # Load in training state
    train_status = BSON.load(joinpath(output_dir, "status.bson"))
    
    # Create model
    model_ctor = mod_lookup(Metalhead, train_status[:model_name])
    model = model_ctor(train_status[:model_args]...; train_status[:model_kwargs]...)

    # Load model parameters
    load_model!(model, joinpath(output_dir, "weights.bson"))

    # Create optimizer (we can't save optimizers yet, as they are currently
    # defiend as closures, so we recreate from scratch every time.)
    opt_ctor = mod_lookup(Flux.Optimise, train_status[:opt_name])
    opt = opt_ctor(params(model), train_status[:opt_args]...; train_status[:opt_kwargs]...)

    # Create train_dataset and val_dataset:
    train_dataset = train_status[:train_dataset_type](train_status[:train_dataset_args]...)
    val_dataset = train_status[:val_dataset_type](val_status[:val_dataset_args]...)

    # Finally use all of this to construct a new TrainState object:
    return TrainState(
        train_status[:model_name],
        train_status[:model_args],
        train_status[:model_kwargs],
        model,

        train_status[:opt_name],
        train_status[:opt_args],
        train_status[:opt_kwargs],
        opt,
        
        train_dataset,
        val_dataset,
        train_status[:max_epochs],

        train_status[:train_loss_history],
        train_status[:val_loss_history],
        train_status[:epoch],
        train_status[:patience],
    )
end

function train_epoch!(ts::TrainState, accelerator = identity)
    # Clear out any previous training loss history
    while length(ts.train_loss_history) < ts.epoch
        push!(ts.train_loss_history, Float64[])
    end
    ts.train_loss_history[ts.epoch] = zeros(Float64, length(ts.train_dataset))

    batch_idx = 1
    avg_batch_time = 0.0
    t_last = time()
    for (x, y) in ts.train_dataset
        t0 = time()
        # Load x/y onto our accelerator, if we have one
        x = accelerator(x)
        y = accelerator(y)
        t1 = time()

        # Push forward pass
        y_hat = ts.model(accelerator(x))
        CUDAnative.synchronize()

        # Calculate loss and backprop (Note the `Flux.Optimise.@interrupts` is
        # just to avoid very large backtraces when interrupting a training by
        # hitting CTRL-C.  It limits the backtrace to this function.)
        t2 = time()
        loss = Flux.crossentropy(y_hat, y)
        Flux.Optimise.@interrupts Flux.back!(loss)
        CUDAnative.synchronize()
        t3 = time()

        # Update weights
        #update!(ts.opt, params(model))
        ts.opt()
        CUDAnative.synchronize()
        t4 = time()

        # Store training loss into loss history
        ts.train_loss_history[ts.epoch][batch_idx] = Flux.Tracker.data(cpu(loss))

        # Update average batch time
        t_now = time()
        avg_batch_time = .99*avg_batch_time + .01*(t_now - t_last)
        t_last = t_now

        # Calculate ETA
        time_left = avg_batch_time*(length(ts.train_dataset) - batch_idx)
        hours = floor(Int,time_left/(60*60))
        minutes = floor(Int, (time_left - hours*60*60)/60)
        seconds = time_left - hours*60*60 - minutes*60
        eta = @sprintf("%dh%dm%ds", hours, minutes, seconds)

        # Show a smoothed loss approximation per-minibatch
        smoothed_loss = mean(ts.train_loss_history[ts.epoch][max(batch_idx-50,1):batch_idx])
        println(@sprintf(
            "[TRAIN %d - %d/%d]: avg loss: %.4f, avg time: %.2fs (%.3fs load, %.3fs fwd, %.3fs back, %.3fs opt), ETA: %s ",
            ts.epoch, batch_idx, length(ts.train_dataset), smoothed_loss,
            avg_batch_time, t1 - t0, t2 - t1, t3 - t2, t4 - t3, eta,
        ))

        batch_idx += 1
    end
end

function validate!(ts::TrainState, accelerator = identity)
    # Get the "fast model", 
    fast_model = Flux.mapleaves(Flux.Tracker.data, ts.model)
    Flux.testmode!(fast_model, true)

    avg_loss = 0
    batch_idx = 1
    for (x, y) in ts.val_dataset
        # move x/y to the accelerator, if necessary
        x = accelerator(x)
        y = accelerator(y)

        # Push x through our fast model and calculate loss
        y_hat = fast_model(x)
        avg_loss += cpu(Flux.crossentropy(y_hat, y))

        print(@sprintf(
            "\r[VAL %d - %d/%d]: %.2f",
            ts.epoch, batch_idx, length(ts.val_dataset), avg_loss/batch_idx,
        ))
        batch_idx += 1
    end
    avg_loss /= length(ts.val_dataset)
    push!(ts.val_loss_history, avg_loss)

    # Return the average loss for this epoch
    return avg_loss
end


function train!(ts::TrainState, output_dir::AbstractString, accelerator = identity)
    # Initialize best_epoch to epoch 0, with Infinity loss
    best_epoch = (0, Inf)

    # Move model and optimizer to accelerator
    #ts.model = accelerator(ts.model)
    #ts.opt = accelerator(ts.opt)

    while ts.epoch < ts.max_epochs
        # Early-stop if we don't improve after `ts.patience` epochs
        if ts.epoch > best_epoch[1] + ts.patience
            @info("Losing patience at epoch $(ts.epoch)!")
            break
        end

        # Train for an epoch
        train_epoch!(ts, accelerator)
        
        # Validate to see how much we've improved
        epoch_loss = validate!(ts, accelerator)

        # Check to see if this epoch is the best we've seen so far
        if epoch_loss < best_epoch[2]
            best_epoch = (ts.epoch, epoch_loss)
        end

        # Save our training state every epoch (but only save the model weights
        # if this was the best epoch yet)
        save_train_state(ts, output_dir, best_epoch[1] == ts.epoch)
        ts.epoch += 1
    end
end
