## SparseTensorDOK: dictionary of keys format

"""
    SparseTensorDOK{Tv,Ti<:Integer,N} <: AbstractSparseTensor{Tv,Ti,N}

`N`-dimensional sparse tensor stored in the **D**ictionary **O**f **K**eys format.
Elements are stored as a dictionary mapping indices (using type `Ti`) to values (of type `Tv`).

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

"""
    SparseTensorDOK{Tv,Ti<:Integer}(undef, dims)
    SparseTensorDOK{Tv,Ti<:Integer,N}(undef, dims)

Construct an uninitialized `N`-dimensional `SparseTensorDOK`
with indices using type `Ti` and elements of type `Tv`.
Here uninitialized means it has no stored entries.

Here `undef` is the `UndefInitializer`. If `N` is supplied,
then it must match the length of `dims`.

# Examples
```julia-repl
julia> A = SparseTensorDOK{Float64, Int8, 3}(undef, (2, 3, 4)) # N given explicitly
2×3×4 SparseTensorDOK{Float64, Int8, 3} with 0 stored entries

julia> B = SparseTensorDOK{Float64, Int8}(undef, (4,)) # N determined by the input
4-element SparseTensorDOK{Float64, Int8, 1} with 0 stored entries

julia> similar(B, 2, 4, 1) # use typeof(B), and the given size
2×4×1 SparseTensorDOK{Float64, Int8, 3} with 0 stored entries
```
"""
SparseTensorDOK{Tv,Ti,N}(::UndefInitializer, dims::Dims{N}) where {Tv,Ti<:Integer,N} =
    SparseTensorDOK(dims, Dict{NTuple{N,Ti},Tv}())
SparseTensorDOK{Tv,Ti}(::UndefInitializer, dims::Dims{N}) where {Tv,Ti<:Integer,N} =
    SparseTensorDOK{Tv,Ti,N}(undef, dims)

## Minimal AbstractArray interface

size(A::SparseTensorDOK) = A.dims

function getindex(A::SparseTensorDOK{Tv,<:Integer,N}, I::Vararg{Int,N}) where {Tv,N}
    @boundscheck checkbounds(A, I...)
    return get(A.dict, I, zero(Tv))
end

function setindex!(A::SparseTensorDOK{Tv,Ti,N}, v, I::Vararg{Int,N}) where {Tv,Ti<:Integer,N}
    @boundscheck checkbounds(A, I...)
    ind, val = convert(NTuple{N,Ti}, I), convert(Tv, v)
    if !iszero(val) || haskey(A.dict, ind)
        A.dict[ind] = val
    end
    return A
end

IndexStyle(::Type{<:SparseTensorDOK}) = IndexCartesian()

## Overloads for specializing outputs

similar(::SparseTensorDOK{<:Any,Ti}, ::Type{Tv}, dims::Dims{N}) where {Tv,Ti<:Integer,N} =
    SparseTensorDOK{Tv,Ti,N}(undef, dims)

## AbstractSparseTensor interface

function dropstored!(f::Function, A::SparseTensorDOK)
    filter!(p -> !f(p.second), A.dict)
    return A
end

numstored(A::SparseTensorDOK) = length(A.dict)
storedindices(A::SparseTensorDOK) = collect(keys(A.dict))
storedvalues(A::SparseTensorDOK) = collect(values(A.dict))
storedpairs(A::SparseTensorDOK) = pairs(A.dict)
