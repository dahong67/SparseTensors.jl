using SparseTensors
using Test
using InteractiveUtils: subtypes

TESTLIST = [
    "AbstractSparseTensor" => "abstract.jl",
    "SparseTensorCOO" => "coo.jl",
    "SparseTensorDOK" => "dok.jl",
]

for (name, path) in TESTLIST
    @testset verbose = true "$name" begin
        include(path)
    end
end
