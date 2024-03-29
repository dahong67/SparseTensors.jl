## SparseTensorDOK: dictionary of keys format

@testitem "constructor" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        dims = (5, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        vals = Tv[1, 0, 10]
        dict = Dict(tuple.(inds...) .=> vals)

        # SparseTensorDOK(dims, dict)
        A = SparseTensorDOK(dims, dict)
        @test typeof(A) === SparseTensorDOK{Tv,Ti,N}
        @test A.dims === dims
        @test A.dict === dict

        # check_Ti(dims, Ti)
        @test_throws ArgumentError SparseTensorDOK((-1, 3, 2)[1:N], dict)
        if Ti !== Int
            @test_throws ArgumentError SparseTensorDOK((Int(typemax(Ti)) + 1, 3, 2)[1:N], dict)
        end
        if N >= 2
            @test_throws ArgumentError SparseTensorDOK((typemax(Int) ÷ 2, 3, 2)[1:N], dict)
        end

        # foreach(ind -> checkbounds_dims(dims, ind...), keys(dict))
        if Ti <: Signed
            badinds = (Ti[2, -1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
            @test_throws ArgumentError SparseTensorDOK(dims, Dict(tuple.(badinds...) .=> vals))
        end
        badinds = (Ti[2, 1, 6], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        @test_throws ArgumentError SparseTensorDOK(dims, Dict(tuple.(badinds...) .=> vals))
    end
end

@testitem "undef constructors" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        dims = (5, 3, 2)[1:N]

        # SparseTensorDOK{Tv,Ti,N}(undef, dims)
        A = SparseTensorDOK{Tv,Ti,N}(undef, dims)
        @test typeof(A) === SparseTensorDOK{Tv,Ti,N}
        @test A.dims === dims
        @test A.dict == Dict{NTuple{N,Ti},Tv}()

        # SparseTensorDOK{Tv,Ti}(undef, dims)
        A = SparseTensorDOK{Tv,Ti}(undef, dims)
        @test typeof(A) === SparseTensorDOK{Tv,Ti,N}
        @test A.dims === dims
        @test A.dict == Dict{NTuple{N,Ti},Tv}()
    end
end

@testitem "AbstractArray constructor" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        dims = (5, 3, 2)[1:N]
        inds = ([2, 1, 4], [1, 3, 2], [1, 2, 1])[1:N]
        inds = sort(tuple.(inds...); by=CartesianIndex)
        vals = Tv[1, 100, 10]
        dict = Dict(inds .=> vals)

        # SparseTensorDOK(Ti, A::Array)
        A = zeros(Tv, dims)
        A[CartesianIndex.(inds)] = vals
        D = SparseTensorDOK(Ti, A)
        @test typeof(D) === SparseTensorDOK{Tv,Ti,N}
        @test D.dims === dims
        @test D.dict == dict

        # SparseTensorDOK(Ti, A::SparseTensorCOO)
        C = SparseTensorCOO(dims, inds, vals)
        D = SparseTensorDOK(Ti, C)
        @test typeof(D) === SparseTensorDOK{Tv,Ti,N}
        @test D.dims === dims
        @test D.dict == dict
    end
end

## Minimal AbstractArray interface

@testitem "size" begin
    @testset "N=$N" for N in 1:3
        dims = (5, 3, 2)[1:N]
        inds = ([2, 1, 4], [1, 3, 2], [1, 2, 1])[1:N]
        vals = [1, 0, 10]
        A = SparseTensorDOK(dims, Dict(tuple.(inds...) .=> vals))

        @test size(A) == dims
        for k in 1:N
            @test size(A, k) == dims[k]
        end
        @test size(A, N + 1) == 1
    end
end

@testitem "getindex" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        dims = (5, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        vals = Tv[1, 0, 10]
        A = SparseTensorDOK(dims, Dict(tuple.(inds...) .=> vals))

        # in bounds
        ind_stored    = (4, 2, 1)[1:N]
        ind_notstored = (3, 2, 2)[1:N]
        @test typeof(A[ind_stored...]) === Tv && A[ind_stored...] == Tv(10)
        @test typeof(A[ind_notstored...]) === Tv && A[ind_notstored...] == zero(Tv)

        # out of bounds
        ind_out1 = (0, 1, 1)[1:N]
        ind_out2 = (6, 3, 2)[1:N]
        @test_throws BoundsError A[ind_out1...]
        @test_throws BoundsError A[ind_out2...]
    end
end

@testitem "setindex!" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        dims = (5, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        vals = Tv[1, 0, 10]

        for val in [-4, 0]
            # store new value in middle
            ind = (3, 2, 2)[1:N]
            A = SparseTensorDOK(dims, Dict(tuple.(inds...) .=> vals))
            A[ind...] = val
            @test typeof(A) === SparseTensorDOK{Tv,Ti,N}
            @test A.dims === dims
            if iszero(val)
                @test A.dict == Dict(tuple.(inds...) .=> vals)
            else
                @test A.dict == Dict([(tuple.(inds...) .=> vals); [ind => val]])
            end

            # store new value at end
            ind = dims
            A = SparseTensorDOK(dims, Dict(tuple.(inds...) .=> vals))
            A[ind...] = val
            @test typeof(A) === SparseTensorDOK{Tv,Ti,N}
            @test A.dims === dims
            if iszero(val)
                @test A.dict == Dict(tuple.(inds...) .=> vals)
            else
                @test A.dict == Dict([(tuple.(inds...) .=> vals); [ind => val]])
            end

            # overwrite existing value
            ind = (4, 2, 1)[1:N]
            A = SparseTensorDOK(dims, Dict(tuple.(inds...) .=> vals))
            A[ind...] = val
            @test typeof(A) === SparseTensorDOK{Tv,Ti,N}
            @test A.dims === dims
            @test A.dict == Dict([(tuple.(inds...) .=> vals)[1:end-1]; [ind => val]])
        end

        # out of bounds
        ind_out1 = (0, 1, 1)[1:N]
        ind_out2 = (6, 3, 2)[1:N]
        A = SparseTensorDOK(dims, Dict(tuple.(inds...) .=> vals))
        @test_throws BoundsError A[ind_out1...] = 0
        @test_throws BoundsError A[ind_out2...] = 0
    end

    # properly handle error during value conversion
    dims = (5, 3, 2)
    inds = sort(tuple.([2, 1, 4], [1, 3, 2], [1, 2, 1]); by = CartesianIndex)
    vals = [1, 0, 10]
    dict = Dict(inds .=> vals)
    A = SparseTensorDOK(dims, copy(dict))
    @test_throws InexactError A[1, 1, 1] = 1.2
    @test A.dict !== dict && A.dict == dict
end

@testitem "IndexStyle" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        @test IndexStyle(SparseTensorDOK{Tv,Ti,N}) === IndexCartesian()
    end
end

## Overloads for specializing outputs

@testitem "similar" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        dims = (5, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        vals = Tv[1, 0, 10]
        A = SparseTensorDOK(dims, Dict(tuple.(inds...) .=> vals))

        # similar(A)
        S = similar(A)
        @test typeof(S) === SparseTensorDOK{Tv,Ti,N}
        @test S.dims === dims
        @test isempty(S.dict)

        # similar(A, ::Type{S})
        for TvNew in [UInt8]
            S = similar(A, TvNew)
            @test typeof(S) === SparseTensorDOK{TvNew,Ti,N}
            @test S.dims === dims
            @test isempty(S.dict)
        end

        # similar(A, dims::Dims)
        for dimsNew in [(2,), (2, 4), (2, 4, 3)]
            S = similar(A, dimsNew)
            @test typeof(S) === SparseTensorDOK{Tv,Ti,length(dimsNew)}
            @test S.dims === dimsNew
            @test isempty(S.dict)
        end

        # similar(A, ::Type{S}, dims::Dims)
        for TvNew in [UInt8], dimsNew in [(2,), (2, 4), (2, 4, 3)]
            S = similar(A, TvNew, dimsNew)
            @test typeof(S) === SparseTensorDOK{TvNew,Ti,length(dimsNew)}
            @test S.dims === dimsNew
            @test isempty(S.dict)
        end
    end
end

## AbstractSparseTensor interface

@testitem "dropstored!" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        dims = (5, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        vals = Tv[1, 0, 10]
        dict = Dict(tuple.(inds...) .=> vals)

        # iszero
        A = SparseTensorDOK(dims, copy(dict))
        dropstored!(iszero, A)
        @test A.dict == Dict((tuple.(inds...) .=> vals)[[1, 3]])

        # beyond tolerance
        A = SparseTensorDOK(dims, copy(dict))
        dropstored!(x -> abs(x) > 5, A)
        @test A.dict == Dict((tuple.(inds...) .=> vals)[1:2])
    end
end

@testitem "numstored / stored[indices|values|pairs]" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        dims = (5, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        inds = tuple.(inds...)
        vals = Tv[1, 0, 10]
        dict = Dict(inds .=> vals)
        A = SparseTensorDOK(dims, dict)

        @test numstored(A) == length(vals)
        dictperm, indsperm = sortperm(collect(keys(dict))), sortperm(inds)
        @test typeof(storedindices(A)) === Vector{NTuple{N,Ti}} && storedindices(A)[dictperm] == inds[indsperm]
        @test typeof(storedvalues(A)) === Vector{Tv} && storedvalues(A)[dictperm] == vals[indsperm]
        @test storedpairs(A) === dict
    end
end
