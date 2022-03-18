function window_partition(x,window_size::Tuple)#Window_size is a tuple of (windowheight,windowwidth),x is a batched image data set
    h,w,c,n=size(x);
    patch_num_h=h//window_size[1];
    patch_num_w=w//window_size[2];
    x=reshape(x,(patch_num_h,window_size[1],patch_num_w,window_size[2],c,n));
    permute!(x, [2,4,5,1,3,6]);
    windows=reshape(x,(window_size[1],window_size[2],c,:))# return a tensor of (window height, window width, channels, num_windows * batchsize)
    return windows 
end

function window_reverse(windows,window_size::Tuple, w, h, c)# the inverse function of window partition
    patch_num_h=h//window_size[1];
    patch_num_w=w//window_size[2];
    x=reshape(windows,(patch_num_y,patch_num_x,window_size[1],window_size[2],c,n));
    permute!(x, [1,3,2,4,5,6]);
    x=reshape(x,(h,w,c,n))
    return x
end