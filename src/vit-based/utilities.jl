using LinearAlgebra: mul!

chunk(A, k::Int; dim::Int = 1) = 
    (selectdim(A, dim, i) for i in Iterators.partition(axes(A,dim), cld(size(A,dim), k)));

pair(t) = (isa(t, Pair)) ? t : (t, t)

function batchmul!(C, A, B)
    for j in axes(A, 4), i in axes(A, 3)
        @views mul!(C[:, :, i, j], A[:, :, i, j], B[:, :, i, j])
    end
    return C
end

function batchmul(A, B)
    T = promote_type(eltype(A), eltype(B))
    C = Array{T}(undef, size(A, 1), size(B)[2:end]...)
    return batchmul!(C, A, B)
end