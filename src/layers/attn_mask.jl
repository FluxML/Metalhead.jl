function get_attn_mask(window_size,shift_size,h,w)
    if shift_size <= 0
        return nothing
    else
        img_mask = zeros(h,w,1,1);
        h_slices = (1:(h-window_size[1]+1),(h-window_size[1]+2):(h-shift_size+1),(h-shift_size+2):h);
        w_slices = (1:(w-window_size[2]+1),(w-window_size[2]+2):(w-shift_size+1),(w-shift_size+2):w);
        cnt = 0;
        for h in h_slices
            for w in w_slices
                img_mask[h,w,:,:].=cnt
                cnt+=1
            end
        end
        mask_windows=window_partition(img_mask,window_size);
        mask_windows=reshape(mask_windows, window_size[1]*window_size[2],:);
        mask_windows=broadcast(-,Flux.unsqueeze(mask_windows,1),Flux.unsqueeze(mask_windows,2));
        function mask_fill(x)
            if x!=0
                return -Inf
            elseif x==0
                return 0.0
            end
        end
        mask_windows=map!(mask_fill,mask_windows,mask_windows);
        return mask_windows
    end
end

