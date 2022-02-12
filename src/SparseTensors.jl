# Top-level file

"""
Support for sparse tensors/arrays. Provides `AbstractSparseTensor` and subtypes.
"""
module SparseTensors

# AbstractArray interface
import Base: size, getindex, setindex!
import Base: IndexStyle

# Overloads for specializing outputs
import Base: show, summary, similar

# Overloads for improving efficiency
import Base: findall, iterate

# Exported types
export AbstractSparseTensor, SparseTensorCOO, SparseTensorDOK

# AbstractSparseTensor interface
export dropstored!, numstored, storedindices, storedvalues, storedpairs

# Exported functions
export indtype

include("abstract.jl")
include("coo.jl")
include("dok.jl")

end
