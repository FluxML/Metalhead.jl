using Distributed, Serialization

mutable struct QueuePool
    # The worker PIDs
    workers::Vector{Int}

    # Channels for communication
    queued_jobs::RemoteChannel
    results::RemoteChannel
    kill_switch::RemoteChannel

    # The ID of the next job to be submitted
    next_job::Int

    # Buffer space where we store results for out-of-order execution
    results_buffer::Dict{Int,Any}
end

function QueuePool(num_workers::Int, proc_func::Function, setup::Expr = :nothing, queue_size=128)
    workers = addprocs(num_workers) #; topology = :master_worker)

    # Tell the workers to include this file and whatever other setup the need,
    # so that they can communicate with us and complete their tasks.
    Distributed.remotecall_eval(Main, workers, quote
        include($(@__FILE__))
        Core.eval(Main, $(setup))
    end)

    # Create our QueuePool
    qp = QueuePool(
        workers,
        RemoteChannel(() -> Channel{Tuple}(queue_size)),
        RemoteChannel(() -> Channel{Tuple}(queue_size)),
        RemoteChannel(() -> Channel{Bool}(1)),
        0,
        Dict{Int,Any}(),
    )
    
    # immediately add a finalizer to it to flip the kill switch and wait for the
    # workers to finish.  EDIT: This doesn't work because we can't switch tasks
    # in a finalizer, apparently.  :(
    #=
    finalizer(close, qp)
    =#

    # Launch workers, running the `worker_task` with a handle to this QueuePool object
    # and the processing function that will be called within the worker loop.
    for id in workers
        Distributed.remote_do(worker_task, id, qp, proc_func)
    end

    # Return QP
    return qp
end

function close(qp::QueuePool)
    # Tell the worker processes to die
    close(qp.queued_jobs)
    put!(qp.kill_switch, true)

    # Wait for the workers to descend into the long, dark sleep
    rmprocs(qp.workers...; waitfor=10)
end

function worker_task(qp::QueuePool, proc_func)
    # Loop unless we're burning this whole queue pool down
    while !isready(qp.kill_switch)
        # Grab the next queued job from the master
        job_id, x = take!(qp.queued_jobs)

        local y
        try
            # Push x through proc_func to get y
            y = proc_func(x)
        catch
            # Just skip bad processing runs
            @warn("Failed to run worker task $(qp.proc_func) on $(x)")
            continue
        end

        # Push the result onto qp.results
        put!(qp.results, (job_id, y))
    end
end


"""
    try_buffer_result!(qp::QueuePool)

Does a nonblocking read of the next result from the QueuePool into our result
buffer.  If no result is available, returns `nothing` immediately.
"""
function try_buffer_result!(qp::QueuePool)
    if isready(qp.results)
        job_id, result = take!(qp.results)
        qp.results_buffer[job_id] = result
        return job_id
    end
    return
end

# Check to see if it's `nothing` and `yield()` if it is.
function try_buffer_result!(qp::QueuePool, t_start::Float64, timeout::Nothing)
    if try_buffer_result!(qp) == nothing
        # No new results available, so just yield
        yield()
    end
end

# Check to see if we've broken through our timeout
function try_buffer_result!(qp::QueuePool, t_start::Float64, timeout::Float64)
    try_buffer_result!(qp, t_start, nothing)

    if (time() - t_start) > timeout
        error("timeout within fetch_result")
    end
end



"""
    push_job!(qp::QueuePool, value)

Push a new job onto the QueuePool, returning the associated job id with this job,
for future usage with `fetch_result(qp, job_id)`
"""
function push_job!(qp::QueuePool, value)
    job_id = qp.next_job
    qp.next_job += 1

    put!(qp.queued_jobs, (job_id, value))
    return job_id
end

"""
    fetch_result(qp::QueuePool; timeout = nothing)

Return a result from the QueuePool, regardless of order.  By default, will wait
for forever; set `timeout` to a value in seconds to time out and throw an error
if a value does not arrive.
"""
function fetch_result(qp::QueuePool; timeout = nothing)
    # If we don't have any results buffered, then pull one in
    t_start = time()
    while isempty(qp.results_buffer)
        try_buffer_result!(qp, t_start, timeout)
    end
    return pop!(qp.results_buffer).second
end

"""
    fetch_result(qp::QueuePool, job_id::Int; timeout = nothing)

Return a result from the QueuePool, in specific order.  By default, will wait
for forever; set `timeout` to a value in seconds to time out and throw an error
if a value does not arrive.
"""
function fetch_result(qp::QueuePool, job_id::Int; timeout=nothing)
    # Keep accumulating results until we get the job_id we're interested in.
    t_start = time()
    while !haskey(qp.results_buffer, job_id)
        try_buffer_result!(qp, t_start, timeout)
    end
    return pop!(qp.results_buffer, job_id)
end
