## SparseTensorDOK: dictionary of keys format

@testset "constructor" begin
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

## Minimal AbstractArray interface

@testset "size" begin
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

@testset "getindex" begin
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

@testset "setindex!" begin
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
            @test A.dict == Dict([(tuple.(inds...) .=> vals); [ind => val]])

            # store new value at end
            ind = dims
            A = SparseTensorDOK(dims, Dict(tuple.(inds...) .=> vals))
            A[ind...] = val
            @test typeof(A) === SparseTensorDOK{Tv,Ti,N}
            @test A.dims === dims
            @test A.dict == Dict([(tuple.(inds...) .=> vals); [ind => val]])

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
end

@testset "IndexStyle" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, Int8]
        @test IndexStyle(SparseTensorDOK{Tv,Ti,N}) === IndexCartesian()
    end
end
