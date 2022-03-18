#Utility function for obtaining relative index and biasin shifted window transformer
#Created by Weihang Xia
using Flux
using Base.Threads
#This function returns a grid of tensor x and y
function meshgrid(x,y)
    xv=first.(Iterators.product(x,y))
    yv=last.(Iterators.product(x,y))
    return [xv,yv]
end

#Input is a 2-Tuple of window_size, the function returns a matrix of the relative position index of an attention window
function get_relative_index(window_size::Tuple)
    coordinate_h=collect(1:window_size[1]);
    coordinate_w=collect(1:window_size[2]);#Initialize height and weight coordinates
    coordinate_pairs=meshgrid(coordinate_h,coordinate_w);
    coordinate_flatten_1=[Array(reshape(transpose(coordinate_pairs[1]),:,1)),Array(reshape(transpose(coordinate_pairs[2]),:,1))];
    coordinate_flatten_2=[Array(reshape(transpose(coordinate_pairs[1]),1,:)),Array(reshape(transpose(coordinate_pairs[2]),1,:))];
    #Flatten the coordinate tensors, and extend their dimension.Corresponding Pytorch codes: coordinate_flatten_1=coordinate_flatten[:, :, None],coordinate_flatten_2=coordinate_flatten[:, None, :]
    coordinate_pairs=nothing;#RAM management.In case the window_size is large
    relative_coordinate=broadcast.(-,coordinate_flatten_1,coordinate_flatten_2);
    coordinate_flatten_1=nothing;
    coordinate_flatten_2=nothing;#RAM management
    relative_index=Matrix{Any}(undef,window_size[1]*window_size[2],window_size[1]*window_size[2]);
    if Flux.CUDA.functional()
        Flux.CUDA.@sync for i in 1:length(relative_coordinate[1])#use Multi-threading technique to boost performance
            relative_index[i]=relative_coordinate[1][i];
            relative_index[i]+=(window_size[1]-1);
            relative_index[i]*=2*window_size[2]-1;
            relative_index[i]+=relative_coordinate[2][i]+window_size[2]-1;
        end
    else
        @threads for i in 1:length(relative_coordinate[1])#use Multi-threading technique to boost performance
            relative_index[i]=relative_coordinate[1][i];
            relative_index[i]+=(window_size[1]-1);
            relative_index[i]*=2*window_size[2]-1;
            relative_index[i]+=relative_coordinate[2][i]+window_size[2]-1;
        end
    end
    return Array(transpose(relative_index))
end

function get_relative_bias(window_size::Tuple,n_heads)
    num_window_elements = window_size[1] * window_size[2];
    relative_position_index=get_relative_index(window_size);
    relative_position_bias=zeros((2 * window_size[1] - 1) * (2 * window_size[2] - 1), n_heads);#Initialize relative bias
    Random.rand!(Distributions.truncated(Normal(0,0.02)), relative_position_bias);#replace zeros with small random numbers
    relative_position_bias=collect(relative_position_bias[i+1,:] for i in relative_position_index);
    relative_position_bias=Flux.stack(relative_position_bias,2);#flatten the tensor
    relative_position_bias=reshape(relative_position_bias,:,num_window_elements,num_window_elements);
    relative_position_bias=Flux.unsqueeze(relative_position_bias,1);#expand dimension
    return relative_position_bias
end