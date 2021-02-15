LearnBase.nobs(data::CuIterator) = nobs(data.batches)
LearnBase.nobs(data::DataLoaders.BufferGetObsParallel) = nobs(data.data)

function train!(data, m, opt; loss, nepochs, schedule, cb)
  local train_loss
  ps = params(m)
  cb = Flux.Optimise.runall(cb)
  nbatches = nobs(data)
  for epoch in 1:nepochs
    opt[2].eta = next!(schedule)
    @info "Epoch $epoch (η = $(opt[2].eta))..."
    for (i, (x, y)) in enumerate(data)
      gs = Flux.gradient(ps) do
        train_loss = loss(x, y, m)
        return train_loss
      end
      (i % 100 == 1) && @info "  Loss ($i / $nbatches) = $train_loss"
      Flux.Optimise.update!(opt, ps, gs)
      cb()
    end
  end
end