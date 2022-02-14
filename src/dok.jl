## SparseTensorDOK: dictionary of keys format

"""
    SparseTensorDOK{Tv,Ti<:Integer,N} <: AbstractSparseTensor{Tv,Ti,N}

Array type for storing sparse tensors in the **D**ictionary **O**f **K**eys format.
Entries are stored as a dictionary mapping indices to values.

Fields:
+ `dims::Dims{N}`               : tuple of dimensions
+ `dict::Dict{NTuple{N,Ti},Tv}` : dictionary mapping indices to values
"""
struct SparseTensorDOK{Tv,Ti<:Integer,N} <: AbstractSparseTensor{Tv,Ti,N}
    dims::Dims{N}                   # Dimensions
    dict::Dict{NTuple{N,Ti},Tv}     # Dictionary

    function SparseTensorDOK{Tv,Ti,N}(dims::Dims{N},
                            dict::Dict{NTuple{N,Ti},Tv}) where {Tv,Ti<:Integer,N}
        check_Ti(dims, Ti)
        foreach(ind -> checkbounds_dims(dims, ind...), keys(dict))
        return new(dims, dict)
    end
end
SparseTensorDOK(dims::Dims{N}, dict::Dict{NTuple{N,Ti},Tv}) where {Tv,Ti<:Integer,N} =
    SparseTensorDOK{Tv,Ti,N}(dims, dict)

## Minimal AbstractArray interface

size(A::SparseTensorDOK) = A.dims

function getindex(A::SparseTensorDOK{Tv,<:Integer,N}, I::Vararg{Int,N}) where {Tv,N}
    @boundscheck checkbounds(A, I...)
    return get(A.dict, I, zero(Tv))
end

function setindex!(A::SparseTensorDOK{Tv,Ti,N}, v, I::Vararg{Int,N}) where {Tv,Ti<:Integer,N}
    @boundscheck checkbounds(A, I...)
    if !iszero(v) || haskey(A.dict, I)
        A.dict[convert(NTuple{N,Ti}, I)] = convert(Tv, v)
    end
    return A
end

IndexStyle(::Type{<:SparseTensorDOK}) = IndexCartesian()

## Overloads for specializing outputs

similar(::SparseTensorDOK{<:Any,Ti}, ::Type{Tv}, dims::Dims{N}) where {Tv,Ti<:Integer,N} =
    SparseTensorDOK(dims, Dict{NTuple{N,Ti},Tv}())

## AbstractSparseTensor interface

function dropstored!(f::Function, A::SparseTensorDOK)
    filter!(p -> !f(p.second), A.dict)
    return A
end

numstored(A::SparseTensorDOK) = length(A.dict)
storedindices(A::SparseTensorDOK) = collect(keys(A.dict))
storedvalues(A::SparseTensorDOK) = collect(values(A.dict))
storedpairs(A::SparseTensorDOK) = pairs(A.dict)
