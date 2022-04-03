## SparseTensorCOO: coordinate format

@testset "constructor" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        dims = (5, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        inds = tuple.(inds...)
        vals = Tv[1, 0, 10]
        perm = sortperm(inds; by = CartesianIndex)
        sinds, svals = inds[perm], vals[perm]

        # SparseTensorCOO(dims, inds, vals) - sorted
        A = SparseTensorCOO(dims, sinds, svals)
        @test typeof(A) === SparseTensorCOO{Tv,Ti,N}
        @test A.dims === dims
        @test A.inds === sinds && A.vals === svals

        # SparseTensorCOO(dims, inds, vals) - unsorted
        A = SparseTensorCOO(dims, inds, vals)
        @test typeof(A) === SparseTensorCOO{Tv,Ti,N}
        @test A.dims === dims
        @test A.inds == sinds && A.vals == svals

        # check_Ti(dims, Ti)
        @test_throws ArgumentError SparseTensorCOO{Tv,Ti,N}((-1, 3, 2)[1:N], sinds, svals)
        if Ti !== Int
            @test_throws ArgumentError SparseTensorCOO{Tv,Ti,N}((Int(typemax(Ti)) + 1, 3, 2)[1:N], sinds, svals)
        end
        if N >= 2
            @test_throws ArgumentError SparseTensorCOO{Tv,Ti,N}((typemax(Int) ÷ 2, 3, 2)[1:N], sinds, svals)
        end

        # check_coo_buffers(inds, vals)
        @test_throws ArgumentError SparseTensorCOO{Tv,Ti,N}(dims, sinds, svals[1:end-1])

        # check_coo_inds(dims, inds) - index in bounds
        if Ti <: Signed
            badinds = (Ti[2, -1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
            badinds = sort(tuple.(badinds...); by = CartesianIndex)
            @test_throws ArgumentError SparseTensorCOO{Tv,Ti,N}(dims, badinds, vals)
        end
        badinds = (Ti[2, 1, 6], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        badinds = sort(tuple.(badinds...); by = CartesianIndex)
        @test_throws ArgumentError SparseTensorCOO{Tv,Ti,N}(dims, badinds, vals)

        # check_coo_inds(dims, inds) - indices sorted
        @test_throws ArgumentError SparseTensorCOO{Tv,Ti,N}(dims, inds, vals)

        # check_coo_inds(dims, inds) - indices unique
        @test_throws ArgumentError SparseTensorCOO{Tv,Ti,N}(dims, [sinds[1:1]; sinds], [svals[1:1]; svals])
    end
end

@testset "undef constructors" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        dims = (5, 3, 2)[1:N]

        # SparseTensorCOO{Tv,Ti,N}(undef, dims)
        A = SparseTensorCOO{Tv,Ti,N}(undef, dims)
        @test typeof(A) === SparseTensorCOO{Tv,Ti,N}
        @test A.dims === dims
        @test A.inds == Vector{NTuple{N,Ti}}()
        @test A.vals == Vector{Tv}()

        # SparseTensorCOO{Tv,Ti}(undef, dims)
        A = SparseTensorCOO{Tv,Ti}(undef, dims)
        @test typeof(A) === SparseTensorCOO{Tv,Ti,N}
        @test A.dims === dims
        @test A.inds == Vector{NTuple{N,Ti}}()
        @test A.vals == Vector{Tv}()
    end
end

@testset "AbstractArray constructor" begin
    @testset "N=$N, Tv=$Tv" for N in 1:3, Tv in [Float64, BigFloat, Int8]
        dims = (5, 3, 2)[1:N]
        inds = ([2, 1, 4], [1, 3, 2], [1, 2, 1])[1:N]
        inds = sort(tuple.(inds...); by=CartesianIndex)
        vals = Tv[1, 100, 10]

        # SparseTensorCOO(A::Array)
        A = zeros(Tv, dims)
        A[CartesianIndex.(inds)] = vals
        C = SparseTensorCOO(A)
        @test typeof(C) === SparseTensorCOO{Tv,Int,N}
        @test C.dims === dims
        @test C.inds == inds && C.vals == vals

        # SparseTensorCOO(A::SparseTensorDOK)
        D = SparseTensorDOK(dims, Dict(inds .=> vals))
        C = SparseTensorCOO(D)
        @test typeof(C) === SparseTensorCOO{Tv,Int,N}
        @test C.dims === dims
        @test C.inds == inds && C.vals == vals

        @testset "Ti=$Ti" for Ti in [Int, UInt8]
            indsTi = convert(Vector{NTuple{N,Ti}}, inds)

            # SparseTensorCOO(Ti, A::Array)
            A = zeros(Tv, dims)
            A[CartesianIndex.(inds)] = vals
            C = SparseTensorCOO(Ti, A)
            @test typeof(C) === SparseTensorCOO{Tv,Ti,N}
            @test C.dims === dims
            @test C.inds == indsTi && C.vals == vals

            # SparseTensorCOO(Ti, A::SparseTensorDOK)
            D = SparseTensorDOK(dims, Dict(inds .=> vals))
            C = SparseTensorCOO(Ti, D)
            @test typeof(C) === SparseTensorCOO{Tv,Ti,N}
            @test C.dims === dims
            @test C.inds == indsTi && C.vals == vals
        end
    end
end

## Minimal AbstractArray interface

@testset "size" begin
    @testset "N=$N" for N in 1:3
        dims = (5, 3, 2)[1:N]
        inds = ([2, 1, 4], [1, 3, 2], [1, 2, 1])[1:N]
        vals = [1, 0, 10]
        A = SparseTensorCOO(dims, tuple.(inds...), vals)

        @test size(A) == dims
        for k in 1:N
            @test size(A, k) == dims[k]
        end
        @test size(A, N + 1) == 1
    end
end

@testset "getindex" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        dims = (6, 3, 2)[1:N]
        inds = (Ti[5, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        vals = Tv[1, 0, 10]
        A = SparseTensorCOO(dims, tuple.(inds...), vals)

        # in bounds
        ind_stored    = (4, 2, 1)[1:N]
        ind_notstored = (3, 2, 2)[1:N]
        @test typeof(A[ind_stored...]) === Tv && A[ind_stored...] == Tv(10)
        @test typeof(A[ind_notstored...]) === Tv && A[ind_notstored...] == zero(Tv)

        # out of bounds
        ind_out1 = (0, 1, 1)[1:N]
        ind_out2 = (7, 3, 2)[1:N]
        @test_throws BoundsError A[ind_out1...]
        @test_throws BoundsError A[ind_out2...]
    end
end

@testset "setindex!" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        dims = (6, 3, 2)[1:N]
        inds = (Ti[5, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        inds = tuple.(inds...)
        vals = Tv[1, 0, 10]
        perm = sortperm(inds; by = CartesianIndex)
        sinds, svals = inds[perm], vals[perm]

        for val in [-4, 0]
            # store new value in middle
            ind = (3, 2, 2)[1:N]
            A = SparseTensorCOO(dims, copy(inds), copy(vals))
            A[ind...] = val
            @test typeof(A) === SparseTensorCOO{Tv,Ti,N}
            @test A.dims === dims
            if iszero(val)
                @test A.inds == sinds && A.vals == svals
            else
                newperm = sortperm([inds; [ind]]; by = CartesianIndex)
                @test A.inds == [inds; [ind]][newperm] && A.vals == [vals; [val]][newperm]
            end

            # store new value at end
            ind = dims
            A = SparseTensorCOO(dims, copy(inds), copy(vals))
            A[ind...] = val
            @test typeof(A) === SparseTensorCOO{Tv,Ti,N}
            @test A.dims === dims
            if iszero(val)
                @test A.inds == sinds && A.vals == svals
            else
                @test A.inds == [sinds; [ind]] && A.vals == [svals; [val]]
            end

            # overwrite existing value
            ind = (4, 2, 1)[1:N]
            A = SparseTensorCOO(dims, copy(inds), copy(vals))
            A[ind...] = val
            @test typeof(A) === SparseTensorCOO{Tv,Ti,N}
            @test A.dims === dims
            @test A.inds == sinds && A.vals == [svals[1:1]; [val]; svals[3:3]]
        end

        # out of bounds
        ind_out1 = (0, 1, 1)[1:N]
        ind_out2 = (7, 3, 2)[1:N]
        A = SparseTensorCOO(dims, inds, vals)
        @test_throws BoundsError A[ind_out1...] = 0
        @test_throws BoundsError A[ind_out2...] = 0
    end

    # properly handle error during value conversion
    dims = (5, 3, 2)
    inds = sort(tuple.([2, 1, 4], [1, 3, 2], [1, 2, 1]); by = CartesianIndex)
    vals = [1, 0, 10]
    A = SparseTensorCOO(dims, copy(inds), copy(vals))
    @test_throws InexactError A[1, 1, 1] = 1.2
    @test A.inds !== inds && A.vals !== vals
    @test A.inds == inds
    @test A.vals == vals
end

@testset "IndexStyle" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        @test IndexStyle(SparseTensorCOO{Tv,Ti,N}) === IndexCartesian()
    end
end

## Overloads for specializing outputs

@testset "similar" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        dims = (5, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        vals = Tv[1, 0, 10]
        A = SparseTensorCOO(dims, tuple.(inds...), vals)

        # similar(A)
        S = similar(A)
        @test typeof(S) === SparseTensorCOO{Tv,Ti,N}
        @test S.dims === dims
        @test isempty(S.inds) && isempty(S.vals)

        # similar(A, ::Type{S})
        for TvNew in [UInt8]
            S = similar(A, TvNew)
            @test typeof(S) === SparseTensorCOO{TvNew,Ti,N}
            @test S.dims === dims
            @test isempty(S.inds) && isempty(S.vals)
        end

        # similar(A, dims::Dims)
        for dimsNew in [(2,), (2, 4), (2, 4, 3)]
            S = similar(A, dimsNew)
            @test typeof(S) === SparseTensorCOO{Tv,Ti,length(dimsNew)}
            @test S.dims === dimsNew
            @test isempty(S.inds) && isempty(S.vals)
        end

        # similar(A, ::Type{S}, dims::Dims)
        for TvNew in [UInt8], dimsNew in [(2,), (2, 4), (2, 4, 3)]
            S = similar(A, TvNew, dimsNew)
            @test typeof(S) === SparseTensorCOO{TvNew,Ti,length(dimsNew)}
            @test S.dims === dimsNew
            @test isempty(S.inds) && isempty(S.vals)
        end
    end
end

## Overloads for improving efficiency

@testset "iterate" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        dims = (5, 3, 2)[1:N]
        inds = (Ti[2, 4], Ti[1, 2], Ti[1, 2])[1:N]
        inds = tuple.(inds...)
        vals = Tv[1, 0]
        A = SparseTensorCOO(dims, inds, vals)

        # iterate(A)
        item, state = iterate(A)
        idxstate, nextptr = state
        @test typeof(item) === Tv && item == zero(Tv)
        @test idxstate == (eachindex(A), eachindex(A)[1])
        @test nextptr == 1

        # iterate(A, state) - stored nonzero
        item, state = iterate(A, state)
        idxstate, nextptr = state
        @test typeof(item) === Tv && item == vals[1]
        @test idxstate == (eachindex(A), eachindex(A)[2])
        @test nextptr == 2

        # iterate(A, state) - unstored entries between stored entries
        for itr in LinearIndices(A)[inds[1]...]+1:LinearIndices(A)[inds[2]...]-1
            item, state = iterate(A, state)
            idxstate, nextptr = state
            @test typeof(item) === Tv && item == zero(Tv)
            @test idxstate == (eachindex(A), eachindex(A)[itr])
            @test nextptr == 2
        end

        # iterate(A, state) - stored zero
        item, state = iterate(A, state)
        idxstate, nextptr = state
        @test typeof(item) === Tv && item == vals[2]
        @test idxstate == (eachindex(A), eachindex(A)[LinearIndices(A)[inds[2]...]])
        @test nextptr == 3

        # iterate(A, state) - unstored entries after stored entries
        for itr in LinearIndices(A)[inds[2]...]+1:length(A)
            item, state = iterate(A, state)
            idxstate, nextptr = state
            @test typeof(item) === Tv && item == zero(Tv)
            @test idxstate == (eachindex(A), eachindex(A)[itr])
            @test nextptr == 3
        end

        # iterate(A, state) - finished
        state = iterate(A, state)
        @test state === nothing
    end
end

## AbstractSparseTensor interface

@testset "dropstored!" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        dims = (5, 3, 2)[1:N]
        inds = (Ti[2, 3, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        inds = tuple.(inds...)
        vals = Tv[1, 0, 10]

        # iszero
        A = SparseTensorCOO(dims, copy(inds), copy(vals))
        dropstored!(iszero, A)
        @test A.inds == inds[[1, 3]] && A.vals == vals[[1, 3]]

        # beyond tolerance
        A = SparseTensorCOO(dims, copy(inds), copy(vals))
        dropstored!(x -> abs(x) > 5, A)
        @test A.inds == inds[1:2] && A.vals == vals[1:2]
    end
end

@testset "numstored / stored[indices|values|pairs]" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        dims = (5, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        inds = tuple.(inds...)
        vals = Tv[1, 0, 10]
        perm = sortperm(inds; by = CartesianIndex)
        sinds, svals = inds[perm], vals[perm]
        A = SparseTensorCOO(dims, inds, vals)

        @test numstored(A) == length(vals)
        @test typeof(storedindices(A)) === Vector{NTuple{N,Ti}} && storedindices(A) == sinds
        @test typeof(storedvalues(A)) === Vector{Tv} && storedvalues(A) == svals
        @test collect(storedpairs(A)) == (sinds .=> svals)
    end
end
