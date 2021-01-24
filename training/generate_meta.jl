using MAT, EzXML

function mat2map(matfile)
  classmap = Dict{String, Int}()
  meta = matread(matfile)["synsets"]

  for (wnid, class, nchild) in zip(meta["WNID"], meta["ILSVRC2012_ID"], meta["num_children"])
    if nchild == 0
        classmap[wnid] = Int(class)
    end
  end

  return classmap
end

function getwnid(file, annotations)
  dir = dirname(file)
  fname = "$(split(basename(file), ".")[1]).xml"
  if isfile(joinpath(annotations, dir, fname))
    xml = readxml(joinpath(annotations, dir, fname))
    wnid = filter(x -> x.name == "name",
                  elements(filter(x -> x.name == "object",
                          elements(root(xml)))[1]))[1].content
  else # no annotation exists, try getting WNID from name
    wnid = split(basename(file), "_")[1]
  end

  return wnid
end

function meta(root, file, classmap, annotations)
  fullpath = joinpath(root, file)
  if isdir(fullpath)
    return map(x -> meta(root, joinpath(file, x), classmap, annotations), readdir(fullpath))
  else
    return (file, classmap[getwnid(file, annotations)])
  end
end

write_metadata(file, xs) = map(x -> write_metadata(file, x), xs)
write_metadata(file, x::Tuple) = write(file, x[1], " ", string(x[2]), "\n")

function generate_metadata(folder, classmap, annotations, output)
  metas = map(x -> meta(folder, x, classmap, annotations), readdir(folder))
  open(output, "w") do io
    write_metadata(io, metas)
  end
end