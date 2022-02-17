## SparseTensorCOO: coordinate format

"""
    SparseTensorCOO{Tv,Ti<:Integer,N} <: AbstractSparseTensor{Tv,Ti,N}

Array type for storing sparse tensors in the **COO**rdinate format.
Entries are stored as a vector of indices and a vector of values.

Fields:
+ `dims::Dims{N}`              : tuple of dimensions
+ `inds::Vector{NTuple{N,Ti}}` : vector of indices
+ `vals::Vector{Tv}`           : vector of values
"""
struct SparseTensorCOO{Tv,Ti<:Integer,N} <: AbstractSparseTensor{Tv,Ti,N}
    dims::Dims{N}                   # Dimensions
    inds::Vector{NTuple{N,Ti}}      # Stored indices
    vals::Vector{Tv}                # Stored values

    function SparseTensorCOO{Tv,Ti,N}(dims::Dims{N}, inds::Vector{NTuple{N,Ti}},
                            vals::Vector{Tv}) where {Tv,Ti<:Integer,N}
        check_Ti(dims, Ti)
        check_coo_buffers(inds, vals)
        check_coo_inds(dims, inds)
        return new(dims, inds, vals)
    end
end
function SparseTensorCOO(dims::Dims{N}, inds::Vector{NTuple{N,Ti}},
                        vals::Vector{Tv}) where {Tv,Ti<:Integer,N}
    if issorted(inds; by = reverse)
        _inds = inds
        _vals = vals
    else
        perm = sortperm(inds; by = reverse)
        _inds = inds[perm]
        _vals = vals[perm]
    end
    SparseTensorCOO{Tv,Ti,N}(dims, _inds, _vals)
end

## Minimal AbstractArray interface

size(A::SparseTensorCOO) = A.dims

function getindex(A::SparseTensorCOO{Tv,<:Integer,N}, I::Vararg{Int,N}) where {Tv,N}
    @boundscheck checkbounds(A, I...)
    ptr = searchsortedfirst(A.inds, I; by = reverse)
    return (ptr == length(A.inds) + 1 || A.inds[ptr] != I) ? zero(Tv) : A.vals[ptr]
end

function setindex!(A::SparseTensorCOO{Tv,Ti,N}, v, I::Vararg{Int,N}) where {Tv,Ti<:Integer,N}
    @boundscheck checkbounds(A, I...)
    ptr = searchsortedfirst(A.inds, I; by = reverse)
    if ptr == length(A.inds) + 1 || A.inds[ptr] != I
        if !iszero(v)
            insert!(A.inds, ptr, convert(NTuple{N,Ti}, I))
            insert!(A.vals, ptr, convert(Tv, v))
        end
    else
        A.vals[ptr] = convert(Tv, v)
    end
    return A
end

IndexStyle(::Type{<:SparseTensorCOO}) = IndexCartesian()

## Overloads for specializing outputs

similar(::SparseTensorCOO{<:Any,Ti}, ::Type{Tv}, dims::Dims{N}) where {Tv,Ti<:Integer,N} =
    SparseTensorCOO(dims, Vector{NTuple{N,Ti}}(), Vector{Tv}())

## Overloads for improving efficiency

# technically specializes the output since the state is different
function iterate(A::SparseTensorCOO{Tv}, state=((eachindex(A),),1)) where {Tv}
    idxstate, nextptr = state
    y = iterate(idxstate...)
    y === nothing && return nothing
    if nextptr > length(A.inds) || A.inds[nextptr] != Tuple(y[1])
        val = zero(Tv)
    else
        val = A.vals[nextptr]
        nextptr += 1
    end
    val, ((idxstate[1], Base.tail(y)...), nextptr)
end

## AbstractSparseTensor interface

function dropstored!(f::Function, A::SparseTensorCOO)
    ptrs = findall(f, A.vals)
    deleteat!(A.inds, ptrs)
    deleteat!(A.vals, ptrs)
    return A
end

numstored(A::SparseTensorCOO) = length(A.vals)
storedindices(A::SparseTensorCOO) = A.inds
storedvalues(A::SparseTensorCOO) = A.vals
storedpairs(A::SparseTensorCOO) = Iterators.map(Pair, A.inds, A.vals)

## AbstractSparseTensor optional interface (internal)

findall_stored(f::Function, A::SparseTensorCOO) =
    [convert(keytype(A), CartesianIndex(ind)) for (ind, val) in storedpairs(A) if f(val)]

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
+ the indices are sorted (`issorted(inds; by=CartesianIndex)`)
+ the indices are all unique (`allunique(inds`)
If not, throw an `ArgumentError`.
"""
function check_coo_inds(dims::Dims{N}, inds::Vector{NTuple{N,Ti}}) where {Ti<:Integer,N}
    # Check all the conditions in a single pass over inds for efficiency
    itr = iterate(inds)
    itr === nothing && return nothing
    prevind, state = itr
    checkbounds_dims(dims, prevind...)
    itr = iterate(inds, state)
    while itr !== nothing
        thisind, state = itr
        if reverse(prevind) < reverse(thisind)
            checkbounds_dims(dims, thisind...)
        elseif reverse(prevind) > reverse(thisind)
            throw(ArgumentError("inds are not sorted"))
        else
            throw(ArgumentError("inds are not all unique"))
        end
        prevind = thisind
        itr = iterate(inds, state)
    end
    return nothing
end
