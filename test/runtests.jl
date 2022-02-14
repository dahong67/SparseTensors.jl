using SparseTensors
using Test
using InteractiveUtils: subtypes

TESTLIST = [
    "AbstractSparseTensor" => "abstract.jl",
    "SparseTensorCOO" => "coo.jl",
    "SparseTensorDOK" => "dok.jl",
]

@testset verbose = true "$name" for (name, path) in TESTLIST
    include(path)
end
