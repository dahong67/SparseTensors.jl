## AbstractSparseTensor type and functions/methods

"""
    AbstractSparseTensor{Tv,Ti<:Integer,N} <: AbstractArray{Tv,N}

Abstract supertype for `N`-dimensional sparse tensors/arrays
with elements of type `Tv` and indices of type `Ti`.
"""
abstract type AbstractSparseTensor{Tv,Ti<:Integer,N} <: AbstractArray{Tv,N} end

## Overloads for specializing outputs

show(io::IO, A::AbstractSparseTensor) = invoke(show, Tuple{IO,Any}, io, A)
function show(io::IO, ::MIME"text/plain", A::AbstractSparseTensor)
    nstored, N = numstored(A), ndims(A)

    # Print summary
    summary(io, A)
    iszero(nstored) && return
    print(io, ":")

    # Print stored entries
    entrylines = get(io, :limit, false) ? displaysize(io)[1] - 4 : typemax(Int)
    pad = map(ndigits, size(A))
    if entrylines >= nstored                    # Enough space to print all the stored entries
        for (ind, val) in storedpairs(A)
            _print_ln_entry(io, pad, ind, val)
        end
    elseif entrylines <= 0                      # No space to print any of the stored entries
        print(io, " \u2026")
    elseif entrylines == 1                      # Only space to print vertical dots
        print(io, '\n', " \u22ee")
    elseif entrylines == 2                      # Only space to print first stored entry
        ind, val = first(storedpairs(A))
        _print_ln_entry(io, pad, ind, val)
        print(io, '\n', ' '^(3 + sum(pad) + 2 * (N - 1) + 3), '\u22ee')
    else                                        # Print the stored entries in two chunks
        # Fetch vectors of entries
        inds, vals = storedindices(A), storedvalues(A)

        # First chunk
        prechunk = div(entrylines - 1, 2, RoundUp)
        for ptr in 1:prechunk
            _print_ln_entry(io, pad, inds[ptr], vals[ptr])
        end

        # Dots
        print(io, '\n', ' '^(3 + sum(pad) + 2 * (N - 1) + 3), '\u22ee')

        # Second chunk
        postchunk = div(entrylines - 1, 2, RoundDown)
        for ptr in nstored-postchunk+1:nstored
            _print_ln_entry(io, pad, inds[ptr], vals[ptr])
        end
    end
end
function _print_ln_entry(io::IO, pad::NTuple{N,Int}, ind::NTuple{N,<:Integer}, val) where {N}
    print(io, '\n', "  [")
    for k in 1:N
        print(io, lpad(Int(ind[k]), pad[k]))
        k == N || print(io, ", ")
    end
    print(io, "]  =  ", val)
end

function summary(io::IO, A::AbstractSparseTensor)
    invoke(summary, Tuple{IO,AbstractArray}, io, A)
    nstored = numstored(A)
    print(io, " with ", nstored, " stored ", nstored == 1 ? "entry" : "entries")
end

## Overloads for improving efficiency

findall_pure(f::Function, A::AbstractSparseTensor) =
    f(zero(eltype(A))) ? invoke(findall, Tuple{Function,AbstractArray}, f, A) : findall_stored(f, A)
findall(f::typeof(!iszero), A::AbstractSparseTensor) = findall_pure(f, A)
findall(f::Base.Fix2{typeof(in)}, A::AbstractSparseTensor) = findall_pure(f, A)

## AbstractSparseTensor interface

"""
    dropstored!(f::Function, A::AbstractSparseTensor{Tv,Ti,N})

Update sparse tensor `A`, dropping stored entries for which `f` is true.
The function `f` is passed one argument: the value of the entry.

For example, use `dropstored!(iszero, A)` to drop all numerical zeros.
"""
function dropstored! end

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

## AbstractSparseTensor optional interface (internal)

"""
    findall_stored(f::Function, A::AbstractSparseTensor)

Variant of `findall(f, A)` that searches over only the stored entries.
Useful when `f(0) == false` is known in advance.
"""
findall_stored(f::Function, A::AbstractSparseTensor) =
    sort!([convert(keytype(A), CartesianIndex(ind)) for (ind, val) in storedpairs(A) if f(val)])

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
function check_Ti(dims::Dims{N}, Ti::Type) where {N}
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
function checkbounds_dims(::Type{Bool}, dims::Dims{N}, I::Vararg{Integer,N}) where {N}
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
function checkbounds_dims(dims::Dims{N}, I::Vararg{Integer,N}) where {N}
    checkbounds_dims(Bool, dims, I...) ||
        throw(ArgumentError("index (= $I) out of bounds (dims = $dims)"))
    return nothing
end
