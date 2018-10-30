using Distributed, Random

function recursive_readdir(root::String)
    ret = String[]
    for (r, dirs, files) in walkdir(root)
        for f in files
            push!(ret, joinpath(r, f)[length(root)+2:end])
        end
    end
    return ret
end


"""
    imagenet_train_data_loader(filename)

Worker thread data loading routine; loads a filename, figures out its label,
and returns the (x, y) pair for later collation.  This is used for training,
and expects data pathnames to look something like `train/nXXX/nXXX_YYYYY.JPEG`
"""
function imagenet_train_data_loader(filename::String)
    synset_mapping = Metalhead.ImageNet.synset_mapping

    # Load image file and preprocess it to get x
    x = Metalhead.imagenet_train_preprocess(filename)

    # Map directory name to class label, then one-hot that
    label = split(basename(filename), "_")[1]
    y = Flux.onehot(synset_mapping[label], 1:length(synset_mapping))[:,:]

    return (x, y)
end

"""
    imagenet_val_data_loader(filename)

Worker thread data loading routine; loads a filename, figures out its label,
and returns the (x, y) pair for later collation.  This is used for validation,
and expects data basenames to look something like `test_XXX.JPEG`.
"""
function imagenet_val_data_loader(filename::String)
    synset_mapping = Metalhead.ImageNet.synset_mapping

    # Load image file and preprocess it to get x
    x = Metalhead.imagenet_val_preprocess(filename)

    # Map filename to class index, then one-hot that
    test_idx = parse(Int, split(splitext(basename(filename))[1], "_")[end])
    label = Metalhead.ImageNet.imagenet_val_labels[test_idx]
    y = Flux.onehot(synset_mapping[label], 1:length(synset_mapping))[:,:]

    return (x, y)
end

struct ImagenetDataset
    # Data we're initialized with
    dataset_root::String
    batch_size::Int
    num_inflight::Int
    data_loader::Function

    # Data we calculate once, at startup
    filenames::Vector{String}
    worker_pool::WorkerPool

    function ImagenetDataset(dataset_root::String, num_workers::Int, batch_size::Int,
                             data_loader::Function = imagenet_val_data_loader,
                             num_inflight::Int = 3)
        # Scan dataset_root for files
        filenames = filter(f -> endswith(f, ".JPEG"), recursive_readdir(dataset_root))

        @assert !isempty(filenames) "Empty dataset folder!"
        @assert num_workers >= 1 "Must have nonnegative integer number of workers!"
        @assert batch_size >= 1 "Must have nonnegative integer batch size!"
        @assert num_inflight >= 1 "Must have nonnegative integer inflight batch size!"

        # Start our worker pool
        @info("Adding $(num_workers) new data workers...")
        worker_pool = WorkerPool(addprocs(num_workers))

        # Have the worker threads load necessary packages like Metalhead, Images, etc...
        @info("Loading worker thread packages...")
        this_file = @__FILE__
        Distributed.remotecall_eval(Main, worker_pool.workers, quote
            using Flux, Images, Metalhead
            include($this_file)
        end)

        return new(dataset_root, batch_size, num_inflight, data_loader, filenames, worker_pool)
    end
end

# Serialize the arguments needed to recreate this ImagenetDataset
function freeze_args(id::ImagenetDataset)
    return (id.dataset_root, length(id.worker_pool.workers), id.batch_size, id.data_loader, id.num_inflight)
end

mutable struct ImagenetIteratorState
    batch_idx::Int
    batches_inflight::Vector{Vector{Future}}
    permutation::Vector{Int64}
    
    function ImagenetIteratorState(id::ImagenetDataset)
        @info("Creating IIS with $(length(id.filenames)) images")
        return new(
            1,
            Vector{Future}[],
            shuffle(1:length(id.filenames)),
        )
    end
end

Base.length(id::ImagenetDataset) = div(length(id.filenames),id.batch_size)
function Base.iterate(id::ImagenetDataset, state=ImagenetIteratorState(id))
    # If we're at the end of this epoch, give up the ghost
    if state.batch_idx + id.batch_size > length(state.permutation) && isempty(state.batches_inflight)
        return nothing
    end

    # Otherwise, read out the next batch (possibly launching more in-flight batches as we go)
    while length(state.batches_inflight) < id.num_inflight && state.batch_idx + id.batch_size <= length(state.permutation)
        # `permutation` randomizes the order in which we access our files, and we take
        # batches out of that randomized order from our list of filenames
        pidxs = state.permutation[state.batch_idx:state.batch_idx+id.batch_size-1]

        # We construct our list of absolute filenames, and submit a series of remote calls
        # to our worker pool, bundling the futures together and storing those into our "inflight"
        # batches.
        batch_filenames = [joinpath(id.dataset_root, id.filenames[idx]) for idx in pidxs]
        next_inflight_batch = remotecall.(Ref(id.data_loader), Ref(id.worker_pool), batch_filenames)
        push!(state.batches_inflight, next_inflight_batch)
        #@show next_inflight_batch

        # Increment our way forward through the epoch
        state.batch_idx += id.batch_size
    end

    # Next, wait for the currently-being-worked-on batch to be done.
    next_inflight_batch = popfirst!(state.batches_inflight)
    #@show next_inflight_batch
    pairs = Tuple[() for idx in 1:id.batch_size]
    while !all(pairs .!= Ref(()))
        # Check to see if any of this inflight batch are ready;
        no_work_done = true
        for idx in 1:id.batch_size
            if pairs[idx] == () && isready(next_inflight_batch[idx])
                no_work_done = false
                pairs[idx] = fetch(next_inflight_batch[idx])
            end
        end

        # If we didn't get any new data this time, sleep for 0.5ms and check for new batches again
        if no_work_done
            sleep(0.0005)
        end
    end

    # Collate X's and Y's into big tensors:
    X = cat((p[1] for p in pairs)...; dims=ndims(pairs[1][1]))
    Y = cat((p[2] for p in pairs)...; dims=ndims(pairs[1][2]))

    # Return the fruit of our labor
    return (X, Y), state
end
