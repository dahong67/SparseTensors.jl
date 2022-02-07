## SparseTensorCOO: coordinate format

"""
    SparseTensorCOO{Tv,Ti<:Integer,N} <: AbstractSparseTensor{Tv,Ti,N}

Array type for storing sparse tensors in the **COO**rdinate format.
Entries are stored as a vector of indices and a vector of values.

Fields:
+ `dims::NTuple{N,Int}`        : tuple of dimensions
+ `inds::Vector{NTuple{N,Ti}}` : vector of indices
+ `vals::Vector{Tv}`           : vector of values
"""
struct SparseTensorCOO{Tv,Ti<:Integer,N} <: AbstractSparseTensor{Tv,Ti,N}
    dims::NTuple{N,Int}             # Dimensions
    inds::Vector{NTuple{N,Ti}}      # Stored indices
    vals::Vector{Tv}                # Stored values

    function SparseTensorCOO{Tv,Ti,N}(dims::NTuple{N,Int}, inds::Vector{NTuple{N,Ti}},
                            vals::Vector{Tv}) where {Tv,Ti<:Integer,N}
        check_Ti(dims, Ti)
        check_coo_buffers(inds, vals)
        check_coo_inds(dims, inds)
        return new(dims, inds, vals)
    end
end
function SparseTensorCOO(dims::NTuple{N,Int}, inds::Vector{NTuple{N,Ti}},
                        vals::Vector{Tv}) where {Tv,Ti<:Integer,N}
    if issorted(inds)
        _inds = inds
        _vals = vals
    else
        perm = sortperm(inds)
        _inds = inds[perm]
        _vals = vals[perm]
    end
    SparseTensorCOO{Tv,Ti,N}(dims, _inds, _vals)
end

## Minimal AbstractArray interface

size(A::SparseTensorCOO) = A.dims

function getindex(A::SparseTensorCOO{Tv,<:Integer,N}, I::Vararg{Int,N}) where {Tv,N}
    @boundscheck checkbounds(A, I...)
    ptr = searchsortedfirst(A.inds, I)
    return (ptr == length(A.inds) + 1 || A.inds[ptr] != I) ? zero(Tv) : A.vals[ptr]
end

function setindex!(A::SparseTensorCOO{Tv,Ti,N}, v, I::Vararg{Int,N}) where {Tv,Ti<:Integer,N}
    @boundscheck checkbounds(A, I...)
    ptr = searchsortedfirst(A.inds, I)
    if ptr == length(A.inds) + 1 || A.inds[ptr] != I
        insert!(A.inds, ptr, convert(NTuple{N,Ti}, I))
        insert!(A.vals, ptr, convert(Tv, v))
    else
        A.vals[ptr] = convert(Tv, v)
    end
    return A
end

IndexStyle(::Type{<:SparseTensorCOO}) = IndexCartesian()

## Overloads for specializing outputs

similar(::SparseTensorCOO{<:Any,Ti}, ::Type{Tv}, dims::Dims{N}) where {Tv,Ti<:Integer,N} =
    SparseTensorCOO(dims, Vector{NTuple{N,Ti}}(), Vector{Tv}())

## AbstractSparseTensor interface

numstored(A::SparseTensorCOO) = length(A.vals)
storedindices(A::SparseTensorCOO) = A.inds
storedvalues(A::SparseTensorCOO) = A.vals
storedpairs(A::SparseTensorCOO) = Iterators.map(Pair, A.inds, A.vals)

## Utilities

"""
    check_coo_buffers(inds, vals)

Check that the `inds` and `vals` buffers are valid:
+ their lengths match (`length(inds) == length(vals)`)
If not, throw an `ArgumentError`.
"""
function check_coo_buffers(inds::Vector{NTuple{N,Ti}}, vals::Vector{Tv}) where {Tv,Ti<:Integer,N}
    length(inds) == length(vals) ||
        throw(ArgumentError("the buffer lengths (length(inds) = $(length(inds)), length(vals) = $(length(vals))) do not match"))
    return nothing
end

"""
    check_coo_inds(dims, inds)

Check that the indices in `inds` are valid:
+ each index is in bounds (`1 ≤ inds[ptr][k] ≤ dims[k]`)
+ the indices are sorted (`issorted(inds)`)
+ the indices are all unique (`allunique(inds`)
If not, throw an `ArgumentError`.
"""
function check_coo_inds(dims::NTuple{N,Int}, inds::Vector{NTuple{N,Ti}}) where {Ti<:Integer,N}
    # Check all the conditions in a single pass over inds for efficiency
    itr = iterate(inds)
    itr === nothing && return nothing
    prevind, state = itr
    checkbounds_dims(dims, prevind...)
    itr = iterate(inds, state)
    while itr !== nothing
        thisind, state = itr
        if prevind < thisind
            checkbounds_dims(dims, thisind...)
        elseif prevind > thisind
            throw(ArgumentError("inds are not sorted"))
        else
            throw(ArgumentError("inds are not all unique"))
        end
        prevind = thisind
        itr = iterate(inds, state)
    end
    return nothing
end
