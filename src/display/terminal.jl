import REPL.Terminals: CSI
const inner_width = 45
const inner_liner = "─"^inner_width

function Base.show(io::IO, frame::PredictionFrame)
    print_frame_table((args...)->nothing, frame)
end

function Base.show(io::IO, frames::Vector{PredictionFrame})
    print_frame_table((args...)->nothing, frames)
end

print_frame_table(image_callback, frame::PredictionFrame) =
    print_frame_table(image_callback, [frame])
function print_frame_table(image_callback, frames::Vector{PredictionFrame})
    lines, cols = displaysize(stdin)
    per_row = div(cols-1, 47)
    remaining = length(frames)
    offset = 0
    buf = IOBuffer()
    io = IOContext(buf, :color => true)
    first_row = min(per_row, remaining)
    println(io, "┌",join(("─"^inner_width for i = 1:first_row),"┬"),"┐")
    while remaining > 0
        this_row = min(per_row, remaining)
        row_frames = frames[(1:this_row) .+ offset]
        if any(p->p.filename !== nothing, row_frames)
            print(io, "│")
            print(io, join(map(row_frames) do p
                sprint() do io
                    fname = p.filename === nothing ? "" : p.filename
                    if length(fname) > inner_width
                        fname = basename(fname)
                    end
                    if length(fname) > inner_width
                        fname = string("…", fname[(end-(inner_width-1)):end])
                    end
                    padding = inner_width - length(fname)
                    print(io, " "^floor(Int, padding/2), fname, " "^ceil(Int, padding/2))
                end
            end, "│"))
            println(io, "│")
        end
        image_callback(io, row_frames)
        println(io, "├",join(("─"^inner_width for i = 1:this_row),"┼"),"┤")
        max_predictions = maximum(map(p->length(p.prediction.sorted_predictions), row_frames))
        any_correct = Symbol[:no for i = 1:this_row]
        for row in 1:max_predictions
            print(io, "│")
            for (col, p) in enumerate(row_frames)
                this_correct = false
                pred = p.prediction.sorted_predictions[row]
                if p.ground_truth === nothing
                    any_correct[col] = :yes
                else
                    ground_truth = p.ground_truth
                    if typeof(pred[1]) != typeof(ground_truth)
                        if method_exists(convert, Tuple{Type{typeof(pred[1])}, typeof(ground_truth)})
                            ground_truth = convert(typeof(pred[1]), ground_truth)
                        else
                            ground_truth = nothing
                        end
                    end
                    if ground_truth !== nothing
                        this_correct = pred[1] == ground_truth
                        this_correct && (any_correct[col] = :yes)
                    else
                        any_correct[col] = :mismatch
                    end
                end
                print_prediction_row(io, pred,
                    annotation = this_correct ? :correct : :none)
                print(io," │")
            end
            println(io)
        end
        if !all(x->x===:yes, any_correct)
            print(io, "│")
            for (col, (had, p)) in enumerate(zip(any_correct, row_frames))
                if had === :yes
                    print(io," "^inner_width, "│")
                else
                    print_prediction_row(io, p.ground_truth=>p.ground_truth_confidence,
                        annotation = had == :mismatch ? :mismatch : :incorrect)
                    print(io," │")
                end
            end
            println(io)
        end
        remaining -= this_row
        offset += this_row
        if remaining != 0
            seps = ['├']
            for i = 1:per_row
                push!(seps, i == per_row ?
                    remaining >= i ? '┤' : '┘' :
                    remaining >= i ? '┼' : '┴')
            end
            println(io, join(seps, inner_liner))
        else
            println(io, "└",join((inner_liner for i = 1:this_row),"┴"),"┘")
        end
        # Write out at the end of each row
        write(stdout, take!(buf))
    end
end

const block_eights = [ Char(0x2590-i) for i = 1:8 ]
function confidence_bar(width, confidence)
    nfull_blocks = trunc(Int, confidence*width)
    neights = trunc(Int, ((confidence*width) - nfull_blocks)*8)
    sprint() do io
        for i = 1:nfull_blocks
            print(io, block_eights[8])
        end
        if neights != 0
            print(io, block_eights[neights])
            nfull_blocks += 1
        end
        print(io, " "^(max(0, width - nfull_blocks)))
    end
end

function print_prediction_row(io::IO, r::Pair; annotation = :none)
    class, confidence = r
    label_text = sprint(print, class)
    label_text = split(label_text, ',')[1]
    # TODO: Maybe use charwidth at some point if we ever have unicode labels
    if length(label_text) > 20
        label_text = string(label_text[1:19],"…")
    end
    color = annotation == :none ? :default :
            annotation == :correct ? :green :
            annotation == :incorrect ? :red :
            annotation == :mismatch ? :yellow :
            error("Unknown annotation")
    printstyled(io, rpad(label_text, 22); color = color)
    print(io, '│', " ")
    if annotation !== :mismatch
        r = round(confidence*100, digits=1)
        printstyled(io, string(
            confidence_bar(15, confidence)),
            lpad(r ≈ 100 ? 100 : r, 4), '%';
            color = color)
    else
        print(io, " "^20)
    end
end

function Base.show(io::IO, p::Prediction)
    for pred in p.sorted_predictions
        print_prediction_row(io, pred)
        println(io)
    end
end
