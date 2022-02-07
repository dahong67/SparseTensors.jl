# Top-level file

"""
Support for sparse tensors/arrays. Provides `AbstractSparseTensor` and subtypes.
"""
module SparseTensors

# AbstractArray interface
import Base: size, getindex, setindex!
import Base: IndexStyle

# Exported types
export AbstractSparseTensor, SparseTensorCOO, SparseTensorDOK

# Exported functions
export indtype

include("abstract.jl")
include("coo.jl")
include("dok.jl")

end
