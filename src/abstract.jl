## AbstractSparseTensor type and functions/methods

"""
    AbstractSparseTensor{Tv,Ti<:Integer,N} <: AbstractArray{Tv,N}

Abstract supertype for `N`-dimensional sparse tensors/arrays
with elements of type `Tv` and indices of type `Ti`.
"""
abstract type AbstractSparseTensor{Tv,Ti<:Integer,N} <: AbstractArray{Tv,N} end

## AbstractSparseTensor interface

"""
    numstored(A::AbstractSparseTensor{Tv,Ti,N})

Return the number of stored entries.
Includes stored numerical zeros; use `count(!iszero,A)`
to count the number of nonzeros.
"""
function numstored end

"""
    storedindices(A::AbstractSparseTensor{Tv,Ti,N})

Return a `Vector{NTuple{N,Ti}}` of all the stored indices.
May share underlying data with `A`.
"""
function storedindices end

"""
    storedvalues(A::AbstractSparseTensor{Tv,Ti,N})

Return a `Vector{Tv}` of all the stored values.
May share underlying data with `A`.
"""
function storedvalues end

"""
    storedpairs(A::AbstractSparseTensor{Tv,Ti,N})

Return an iterator over index => value pairs for all the stored entries.
May share underlying data with `A`.
"""
function storedpairs end

## Generic methods

"""
    indtype(T::Type{<:AbstractSparseTensor})
    indtype(A::AbstractSparseTensor)

Return the index type of a sparse tensor.

# Examples
```julia-repl
julia> indtype(AbstractSparseTensor{Float64,Int8,2})
Int8
```
"""
indtype(::Type{<:AbstractSparseTensor}) = Integer
indtype(::Type{<:AbstractSparseTensor{<:Any,Ti}}) where {Ti<:Integer} = Ti
indtype(A::AbstractSparseTensor) = indtype(typeof(A))

## Utilities

"""
    check_Ti(dims, Ti)

Check that the `dims` tuple and `Ti` index type are valid:
+ `dims` are nonnegative and fit in `Ti` (`0 ≤ dims[k] ≤ typemax(Ti)`)
+ corresponding length fits in `Int` (`prod(dims) ≤ typemax(Int)`)
If not, throw an `ArgumentError`.
"""
function check_Ti(dims::NTuple{N,Int}, Ti::Type) where {N}
    # Check that dims are nonnegative and fit in Ti
    maxTi = typemax(Ti)
    for k in 1:N
        dim = dims[k]
        dim >= 0 || throw(ArgumentError("the size along dimension $k (dims[$k] = $dim) is negative"))
        dim <= maxTi ||
            throw(ArgumentError("the size along dimension $k (dims[$k] = $dim) does not fit in Ti = $(Ti)"))
    end

    # Check that corresponding length fits in Int
    len = reduce(widemul, dims)
    len <= typemax(Int) ||
        throw(ArgumentError("number of elements (length = $len) does not fit in Int (prevents linear indexing)"))
    # do not need to check that dims[k] <= typemax(Int) for CartesianIndex since eltype(dim) == Int

    return nothing
end

"""
    checkbounds_dims(Bool, dims, I...)

Return `true` if the specified indices `I` are in bounds for an array
with the given dimensions `dims`. Useful for checking the inputs to constructors.
"""
function checkbounds_dims(::Type{Bool}, dims::NTuple{N,Int}, I::Vararg{Integer,N}) where {N}
    for k in 1:N
        (1 <= I[k] <= dims[k]) || return false
    end
    return true
end

"""
    checkbounds_dims(dims, I...)

Throw an error if the specified indices `I` are not in bounds for an array
with the given dimensions `dims`. Useful for checking the inputs to constructors.
"""
function checkbounds_dims(dims::NTuple{N,Int}, I::Vararg{Integer,N}) where {N}
    checkbounds_dims(Bool, dims, I...) ||
        throw(ArgumentError("index (= $I) out of bounds (dims = $dims)"))
    return nothing
end
