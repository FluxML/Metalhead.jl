module PascalVOC

import ..Metalhead
import ..Metalhead: testimgs
import ..Metalhead: ValidationImage, TrainingImage, ValData, TestData, TrainData, ObjectClass, ObjectCoords, labels
import ..Metalhead: ValidationBoxImage, TrainingBoxImage, ValBoxData, TrainBoxData

abstract type DataSet <: Metalhead.DataSet end
using Images
using ColorTypes
using ColorTypes: N0f8

using LightXML

BASE_PATH = joinpath(@__DIR__,"..","..","datasets","VOC2012/")
ANNOTATION_PATH = joinpath(@__DIR__,"..","..","datasets","VOC2012","Annotations/")

const PascalVOCLabels = ["person","bird", "cat", "cow", "dog", "horse", "sheep",
		   		   "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
  		   		   "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]
labels2id = Dict([(key,i) for (i,key) in enumerate(PascalVOCLabels)])
const PascalVOCFileNames = []

function check_populate_xml_file_names()
	if(isdir(BASE_PATH) && length(PascalVOCFileNames) < 1)
		for r in readdir(ANNOTATION_PATH)
	    	push!(PascalVOCFileNames,r)
		end
	end
end

# Utility function
function parse_img_xml2jpeg(c::String)
    # Remove xml and add JPEG 
    c = string(c[1:end-4],".jpg")
    c = string(BASE_PATH,"JPEGImages/",c)
    img = load(c)
    return img
end

struct PVocClass <: ObjectClass
    class::Int
end

struct PVocCoord <: ObjectCoords
    xmin::Int
    ymin::Int
    xmax::Int
    ymax::Int
end

labels(::Type{PVocClass}) = PascalVOCLabels

struct RawFS <: DataSet
    folder::String
end

function get_class_coords(c::String)
	xdoc = parse_file(string(ANNOTATION_PATH,c))
    xroot = root(xdoc)
    objects = xroot["object"]

    classes = Vector{PVocClass}()
    coords = Vector{PVocCoord}()

    for obj in objects
    	push!(classes,PVocClass(labels2id[content(obj["name"][1])]))
    	push!(coords,PVocCoord(parse(Int32,content(obj["bndbox"][1]["xmin"][1])),
    							parse(Int32,content(obj["bndbox"][1]["ymin"][1])),
    							parse(Int32,content(obj["bndbox"][1]["xmax"][1])),
    							parse(Int32,content(obj["bndbox"][1]["ymax"][1]))))
    end

    return classes,coords
end

# Base.size(v::ValData{<:DataSet}) = (10000,)

Base.size(v::TrainBoxData{<:DataSet}) = (length(PascalVOCFileNames),)

testimgs(::DataSet) = error("Test Set Not Yet Implemented")

function Base.getindex(v::TrainBoxData{RawFS}, i::Integer)
    TrainingBoxImage(DataSet,i,parse_img_xml2jpeg(PascalVOCFileNames[i]),
    	get_class_coords(PascalVOCFileNames[i])...)
end

end
