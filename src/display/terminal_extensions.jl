using TerminalExtensions

function Base.display(disp::TerminalExtensions.iTerm2.InlineDisplay, p::PredictionFrame)
    Base.display(disp, [p])
end

function Base.display(disp::TerminalExtensions.iTerm2.InlineDisplay, frames::Vector{PredictionFrame})
    print_frame_table(image_display_callback, frames)
end

function image_display_callback(buf, row_frames)
    aspect_ratios = [size(p.img, 1) / size(p.img, 2) for p in row_frames]
    # An extra 17/7 which seems to be the default aspect ratio of
    # an iterm terminal cell.
    max_image_height = maximum(round.(Int, aspect_ratios .* (inner_width/(17/7))))
    for i = 1:max_image_height
        print(buf, "│")
        for i = 1:length(row_frames)
            print(buf, CSI, string(inner_width), 'C', "│")
        end
        println(buf)
    end
    print(buf, CSI, string(1), 'A')
    print(buf, CSI, string(1), 'C')
    for p in row_frames
        print(buf, CSI, string(max_image_height-1), 'A')
        display_img(buf, p.img, height=string(max_image_height), width = string(inner_width))
    end
    println(buf)
end

function display_img(io::IO, img; kwargs...)
    buf = IOBuffer()
    show(buf,MIME"image/png"(),img)
    TerminalExtensions.iTerm2.display_file(take!(buf); io=io,filename="image",inline=true,preserveAspectRatio=true,kwargs...)
end
