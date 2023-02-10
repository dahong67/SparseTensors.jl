using SparseTensors
using Test
using InteractiveUtils: subtypes

TESTLIST = [
    "AbstractSparseTensor" => "items/abstract.jl",
    "SparseTensorCOO" => "items/coo.jl",
    "SparseTensorDOK" => "items/dok.jl",
]

@testset verbose = true "$name" for (name, path) in TESTLIST
    include(path)
end
