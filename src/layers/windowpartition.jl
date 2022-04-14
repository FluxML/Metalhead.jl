function window_partition(x,window_size::Tuple)#Window_size is a tuple of (windowheight,windowwidth),x is a batched image data set
    h,w,c,b=size(x);
    patch_num_h=convert(Int64,h/window_size[1]);
    patch_num_w=convert(Int64,w/window_size[2]);
    x=reshape(x,window_size[1],patch_num_h,window_size[2],patch_num_w,c,n);
    x=permutedims(x, [1,3,5,2,4,6]);
    x=reshape(x,window_size[1],window_size[2],c,:)# return a tensor of (window height, window width, channels,num_windows * batchsize)
    x=permutedims(x,[4,3,1,2])
    return x #nw*b,c,h,w
end

function window_reverse(windows,window_size::Tuple, h, w, c, b)# the inverse function of window partition
    x=permutedims(x,invperm([4,3,1,2]))
    patch_num_h=convert(Int64,h/window_size[1]);
    patch_num_w=convert(Int64,w/window_size[2]);
    x=reshape(windows,window_size[1],window_size[2],c,patch_num_h,patch_num_w,b);
    x=permutedims(x,invperm([1,3,5,2,4,6]));
    x=reshape(x,window_size[1],patch_num_h,window_size[2],patch_num_w,c,b)
    x=reshape(x,h,w,c,b)
    return x #h,w,c,b
end