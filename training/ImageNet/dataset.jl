using Distributed, Random, Printf

include("queuepool.jl")

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
    t_start = time()
    synset_mapping = Metalhead.ImageNet.synset_mapping

    # Load image file and preprocess it to get x
    x, dt0, dt1, dt2 = Metalhead.imagenet_train_preprocess(filename)

    # Map directory name to class label, then one-hot that
    label = split(basename(filename), "_")[1]
    y = Flux.onehot(synset_mapping[label], 1:length(synset_mapping))[:,:]

    #println(@sprintf("[%.03fs, %.03fs, %.03fs]: %s", dt0, dt1, dt2, filename))
    return (x, y)
end

"""
    imagenet_val_data_loader(filename)

Worker thread data loading routine; loads a filename, figures out its label,
and returns the (x, y) pair for later collation.  This is used for validation,
and expects data basenames to look something like `test_XXX.JPEG`.
"""
function imagenet_val_data_loader(filename::String)
    t_start = time()
    synset_mapping = Metalhead.ImageNet.synset_mapping

    # Load image file and preprocess it to get x
    x = Metalhead.imagenet_val_preprocess(filename)

    # Map filename to class index, then one-hot that
    test_idx = parse(Int, split(splitext(basename(filename))[1], "_")[end])
    label = Metalhead.ImageNet.imagenet_val_labels[test_idx]
    y = Flux.onehot(synset_mapping[label], 1:length(synset_mapping))[:,:]

    println(@sprintf("%s: %.3fs", filename, time() - t_start)) 
    return (x, y)
end

struct ImagenetDataset
    # Data we're initialized with
    dataset_root::String
    batch_size::Int
    data_loader::Function

    # Data we calculate once, at startup
    filenames::Vector{String}
    queue_pool::QueuePool

    function ImagenetDataset(dataset_root::String, num_workers::Int, batch_size::Int,
                             data_loader::Function = imagenet_val_data_loader)
        # Scan dataset_root for files
        filenames = filter(f -> endswith(f, ".JPEG"), recursive_readdir(dataset_root))

        @assert !isempty(filenames) "Empty dataset folder!"
        @assert num_workers >= 1 "Must have nonnegative integer number of workers!"
        @assert batch_size >= 1 "Must have nonnegative integer batch size!"

        # Start our worker pool
        @info("Adding $(num_workers) new data workers...")
        queue_pool = QueuePool(num_workers, data_loader, quote
            # The workers need to be able to load images and preprocess them via Metalhead
            using Flux, Images, Metalhead
            include($(@__FILE__))
        end)

        return new(dataset_root, batch_size, data_loader, filenames, queue_pool)
    end
end

# Serialize the arguments needed to recreate this ImagenetDataset
function freeze_args(id::ImagenetDataset)
    return (id.dataset_root, length(id.queue_pool.workers), id.batch_size, id.data_loader)
end
Base.length(id::ImagenetDataset) = div(length(id.filenames),id.batch_size)

mutable struct ImagenetIteratorState
    batch_idx::Int
    job_offset::Int
    
    function ImagenetIteratorState(id::ImagenetDataset)
        @info("Creating IIS with $(length(id.filenames)) images")

        # Build permutation for this iteration
        permutation = shuffle(1:length(id.filenames))

        # Push first job, save value to get job_offset (we know that all jobs
        # within this iteration will be consequtive, so we only save the offset
        # of the first one, and can use that to determine the job ids of every
        # subsequent job:
        filename = joinpath(id.dataset_root, id.filenames[permutation[1]])
        job_offset = push_job!(id.queue_pool, filename)

        # Next, push every other job
        for pidx in permutation[2:end]
            filename = joinpath(id.dataset_root, id.filenames[pidx])
            push_job!(id.queue_pool, filename)
        end
        return new(
            0,
            job_offset,
        )
    end
end

function Base.iterate(id::ImagenetDataset, state=ImagenetIteratorState(id))
    # If we're at the end of this epoch, give up the ghost
    if state.batch_idx > length(id)
        return nothing
    end

    # Otherwise, wait for the next batch worth of jobs to finish on our queue pool
    next_batch_job_ids = state.job_offset .+ (0:(id.batch_size-1)) .+ id.batch_size*state.batch_idx
    # Next, wait for the currently-being-worked-on batch to be done.
    pairs = fetch_result.(Ref(id.queue_pool), next_batch_job_ids)
    state.batch_idx += 1

    # Collate X's and Y's into big tensors:
    X = cat((p[1] for p in pairs)...; dims=ndims(pairs[1][1]))
    Y = cat((p[2] for p in pairs)...; dims=ndims(pairs[1][2]))

    # Return the fruit of our labor
    return (X, Y), state
end
