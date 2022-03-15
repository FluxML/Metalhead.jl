#Utility function for obtaining relative index in shifted window transformer
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
    relative_index=Matrix{Any}(undef,window_size[1]*window_size[2],window_size[1]*window_size[2])
    @threads for i in 1:length(relative_coordinate[1])#use Multi-threading technique to boost performance
        relative_index[i]=relative_coordinate[1][i];
        relative_index[i]+=(window_size[1]-1);
        relative_index[i]*=2*window_size[2]-1;
        relative_index[i]+=relative_coordinate[2][i]+window_size[2]-1;
    end
    return relative_index
end