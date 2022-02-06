using SparseTensors
using Test

TESTLIST = [
    "SparseTensorCOO" => "coo.jl",
    "SparseTensorDOK" => "dok.jl",
]

for (name, path) in TESTLIST
    @testset "$name" begin
        include(path)
    end
end
