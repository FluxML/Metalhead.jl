struct DropPath
    droprate
    distribution
end

function DropPath(drop)
    droprate=drop;
    distribution=Bernoulli(droprate);
    DropPath(drop,distribution);
end

@functor DropPath ()

function (dp::DropPath)(x::Array)
    if dp.droprate==0
        return x
    else
        keep_rate=1-dp.droprate;
        shape=ones(Int,length(size(x)));
        shape[1]=Int(size(x)[1]);
        shape=Tuple(shape);
        mask=rand(dp.distribution,shape);
        mask=convert(Array{Int},mask)
        return broadcast(*,x,mask)
    end
end