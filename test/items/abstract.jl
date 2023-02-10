## AbstractSparseTensor type and functions/methods

## Overloads for specializing outputs

@testitem "show(io, A)" begin
    @testset "nstored=$nstored, N=$N, Ti=$Ti, Tv=$Tv" for nstored in 0:3, N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, UInt8]
        dims = (10, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        inds = tuple.(inds...)
        vals = Tv[0, 1, 10]

        # Take subset of entries
        inds = inds[1:nstored]
        vals = vals[1:nstored]

        # SparseTensorCOO
        perm = sortperm(inds; by = CartesianIndex)
        sinds, svals = inds[perm], vals[perm]
        C = SparseTensorCOO(dims, inds, vals)
        @test sprint(show, C; context=:module=>@__MODULE__) == "SparseTensorCOO{$Tv, $Ti, $N}($dims, $sinds, $svals)"

        # SparseTensorDOK
        dict = Dict(inds .=> vals)
        D = SparseTensorDOK(dims, dict)
        @test sprint(show, D; context=:module=>@__MODULE__) == "SparseTensorDOK{$Tv, $Ti, $N}($dims, $dict)"
    end
end

@testitem "show(io, ::MIME\"text/plain\", A)" begin
    @testset "nstored=$nstored, N=$N, Ti=$Ti, Tv=$Tv" for nstored in 0:3, N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, UInt8]
        dims = (10, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        inds = tuple.(inds...)
        vals = Tv[0, 1, 10]

        # Take subset of entries
        inds = inds[1:nstored]
        vals = vals[1:nstored]

        # SparseTensorCOO
        perm = sortperm(inds; by = CartesianIndex)
        sinds, svals = inds[perm], vals[perm]
        C = SparseTensorCOO(dims, inds, vals)
        entrystrs = map(sinds, svals) do ind, val
            indstr = join(lpad.(Int.(ind), ndigits.(dims)), ", ")
            return "  [$indstr]  =  $val"
        end
        showstr = length(vals) == 0 ? summary(C) : "$(summary(C)):\n$(join(entrystrs,'\n'))"
        @test sprint(show, MIME("text/plain"), C) == showstr

        # SparseTensorDOK
        dict = Dict(inds .=> vals)
        D = SparseTensorDOK(dims, dict)
        entrystrs = map(collect(dict)) do (ind, val)
            indstr = join(lpad.(Int.(ind), ndigits.(dims)), ", ")
            return "  [$indstr]  =  $val"
        end
        showstr = length(vals) == 0 ? summary(D) : "$(summary(D)):\n$(join(entrystrs,'\n'))"
        @test sprint(show, MIME("text/plain"), D) == showstr
    end

    @testset "displayheight=$displayheight" for displayheight in 0:11
        iocontext = IOContext(IOBuffer(), :displaysize => (displayheight, 80), :limit => true)
        dims = (5, 10, 2)
        inds = tuple.(
            UInt8[4, 1, 2, 5, 5, 3, 2],
            UInt8[1, 2, 3, 4, 10, 7, 6],
            UInt8[1, 1, 2, 1, 2, 1, 2],
        )
        vals = Float32[0.0, 0.5, 0.4, 0.2, 0.3, 0.8, 0.9]

        # SparseTensorCOO
        showstr = Dict(
            0 => "5×10×2 SparseTensorCOO{Float32, UInt8, 3} with 7 stored entries: \u2026",
            1 => "5×10×2 SparseTensorCOO{Float32, UInt8, 3} with 7 stored entries: \u2026",
            2 => "5×10×2 SparseTensorCOO{Float32, UInt8, 3} with 7 stored entries: \u2026",
            3 => "5×10×2 SparseTensorCOO{Float32, UInt8, 3} with 7 stored entries: \u2026",
            4 => "5×10×2 SparseTensorCOO{Float32, UInt8, 3} with 7 stored entries: \u2026",
            5 => """
            5×10×2 SparseTensorCOO{Float32, UInt8, 3} with 7 stored entries:
             \u22ee""",
            6 => """
            5×10×2 SparseTensorCOO{Float32, UInt8, 3} with 7 stored entries:
              [4,  1, 1]  =  0.0
                          \u22ee""",
            7 => """
            5×10×2 SparseTensorCOO{Float32, UInt8, 3} with 7 stored entries:
              [4,  1, 1]  =  0.0
                          \u22ee
              [5, 10, 2]  =  0.3""",
            8 => """
            5×10×2 SparseTensorCOO{Float32, UInt8, 3} with 7 stored entries:
              [4,  1, 1]  =  0.0
              [1,  2, 1]  =  0.5
                          \u22ee
              [5, 10, 2]  =  0.3""",
            9 => """
            5×10×2 SparseTensorCOO{Float32, UInt8, 3} with 7 stored entries:
              [4,  1, 1]  =  0.0
              [1,  2, 1]  =  0.5
                          \u22ee
              [2,  6, 2]  =  0.9
              [5, 10, 2]  =  0.3""",
            10 => """
            5×10×2 SparseTensorCOO{Float32, UInt8, 3} with 7 stored entries:
              [4,  1, 1]  =  0.0
              [1,  2, 1]  =  0.5
              [5,  4, 1]  =  0.2
                          \u22ee
              [2,  6, 2]  =  0.9
              [5, 10, 2]  =  0.3""",
            11 => """
            5×10×2 SparseTensorCOO{Float32, UInt8, 3} with 7 stored entries:
              [4,  1, 1]  =  0.0
              [1,  2, 1]  =  0.5
              [5,  4, 1]  =  0.2
              [3,  7, 1]  =  0.8
              [2,  3, 2]  =  0.4
              [2,  6, 2]  =  0.9
              [5, 10, 2]  =  0.3""",
        )[displayheight]
        C = SparseTensorCOO(dims, inds, vals)
        @test sprint(show, MIME("text/plain"), C; context = IOContext(iocontext, :module=>@__MODULE__)) == showstr

        # SparseTensorDOK
        showstr = Dict(
            0 => "5×10×2 SparseTensorDOK{Float32, UInt8, 3} with 7 stored entries: \u2026",
            1 => "5×10×2 SparseTensorDOK{Float32, UInt8, 3} with 7 stored entries: \u2026",
            2 => "5×10×2 SparseTensorDOK{Float32, UInt8, 3} with 7 stored entries: \u2026",
            3 => "5×10×2 SparseTensorDOK{Float32, UInt8, 3} with 7 stored entries: \u2026",
            4 => "5×10×2 SparseTensorDOK{Float32, UInt8, 3} with 7 stored entries: \u2026",
            5 => """
            5×10×2 SparseTensorDOK{Float32, UInt8, 3} with 7 stored entries:
             \u22ee""",
            6 => """
            5×10×2 SparseTensorDOK{Float32, UInt8, 3} with 7 stored entries:
              [1,  2, 1]  =  0.5
                          \u22ee""",
            7 => """
            5×10×2 SparseTensorDOK{Float32, UInt8, 3} with 7 stored entries:
              [1,  2, 1]  =  0.5
                          \u22ee
              [5,  4, 1]  =  0.2""",
            8 => """
            5×10×2 SparseTensorDOK{Float32, UInt8, 3} with 7 stored entries:
              [1,  2, 1]  =  0.5
              [2,  3, 2]  =  0.4
                          \u22ee
              [5,  4, 1]  =  0.2""",
            9 => """
            5×10×2 SparseTensorDOK{Float32, UInt8, 3} with 7 stored entries:
              [1,  2, 1]  =  0.5
              [2,  3, 2]  =  0.4
                          \u22ee
              [5, 10, 2]  =  0.3
              [5,  4, 1]  =  0.2""",
            10 => """
            5×10×2 SparseTensorDOK{Float32, UInt8, 3} with 7 stored entries:
              [1,  2, 1]  =  0.5
              [2,  3, 2]  =  0.4
              [3,  7, 1]  =  0.8
                          \u22ee
              [5, 10, 2]  =  0.3
              [5,  4, 1]  =  0.2""",
            11 => """
            5×10×2 SparseTensorDOK{Float32, UInt8, 3} with 7 stored entries:
              [1,  2, 1]  =  0.5
              [2,  3, 2]  =  0.4
              [3,  7, 1]  =  0.8
              [2,  6, 2]  =  0.9
              [4,  1, 1]  =  0.0
              [5, 10, 2]  =  0.3
              [5,  4, 1]  =  0.2""",
        )[displayheight]
        D = SparseTensorDOK(dims, Dict(inds .=> vals))
        @test sprint(show, MIME("text/plain"), D; context = IOContext(iocontext, :module=>@__MODULE__)) == showstr
    end
end

@testitem "summary(io, A)" begin
    @testset "nstored=$nstored, N=$N, Ti=$Ti, Tv=$Tv" for nstored in 0:3, N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, UInt8]
        dims = (10, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        inds = tuple.(inds...)
        vals = Tv[0, 1, 10]

        # Take subset of entries
        inds = inds[1:nstored]
        vals = vals[1:nstored]

        # References
        dimstr = ["10-element", "10×3", "10×3×2"]
        valstr = "with $(length(vals)) stored " * (length(vals) == 1 ? "entry" : "entries")

        # SparseTensorCOO
        C = SparseTensorCOO(dims, inds, vals)
        @test sprint(summary, C; context=:module=>@__MODULE__) == "$(dimstr[N]) SparseTensorCOO{$Tv, $Ti, $N} $valstr"

        # SparseTensorDOK
        D = SparseTensorDOK(dims, Dict(inds .=> vals))
        @test sprint(summary, D; context=:module=>@__MODULE__) == "$(dimstr[N]) SparseTensorDOK{$Tv, $Ti, $N} $valstr"
    end
end

## Overloads for improving efficiency

@testitem "findall(A)" begin
    @testset "N=$N, Ti=$Ti" for N in 1:3, Ti in [Int, UInt8]
        dims = (10, 3, 2)[1:N]
        inds = (Ti[2, 5, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        inds = tuple.(inds...)
        vals = Bool[1, 0, 1]

        C = SparseTensorCOO(dims, inds, vals)
        D = SparseTensorDOK(dims, Dict(inds .=> vals))
        A = collect(C)

        @test typeof(findall(C)) == typeof(findall(D)) == typeof(findall(A))
        @test findall(C) == findall(D) == findall(A)
    end
end

@testitem "findall(f, A)" begin
    @testset "N=$N, Ti=$Ti, Tv=$Tv" for N in 1:3, Ti in [Int, UInt8], Tv in [Float64, BigFloat, UInt8]
        dims = (10, 3, 2)[1:N]
        inds = (Ti[2, 5, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        inds = tuple.(inds...)
        vals = Tv[0, 1, 10]

        # Form tensors
        C = SparseTensorCOO(dims, inds, vals)
        D = SparseTensorDOK(dims, Dict(inds .=> vals))
        A = collect(C)

        # findall(!iszero, A)
        @test typeof(findall(!iszero, C)) == typeof(findall(!iszero, D)) == typeof(findall(!iszero, A))
        @test findall(!iszero, C) == findall(!iszero, D) == findall(!iszero, A)

        # findall(in(...), A) - f(0) == false
        @test typeof(findall(in(Tv[1, 2]), C)) == typeof(findall(in(Tv[1, 2]), D)) == typeof(findall(in(Tv[1, 2]), A))
        @test findall(in(Tv[1, 2]), C) == findall(in(Tv[1, 2]), D) == findall(in(Tv[1, 2]), A)

        # findall(in(...), A) - f(0) == true
        @test typeof(findall(in(Tv[0, 1]), C)) == typeof(findall(in(Tv[0, 1]), D)) == typeof(findall(in(Tv[0, 1]), A))
        @test findall(in(Tv[0, 1]), C) == findall(in(Tv[0, 1]), D) == findall(in(Tv[0, 1]), A)
    end
end

## Generic methods

@testitem "indtype" begin
    using InteractiveUtils: subtypes

    # indtype(T::Type{<:AbstractSparseTensor})
    @testset "T::Type{<:AbstractSparseTensor}" begin
        AT = AbstractSparseTensor
        @test indtype(AT) == Integer
        for T in subtypes(AT)
            @test indtype(T) == Integer
        end
    end
    @testset "T::Type{<:AbstractSparseTensor{$Tv}}" for Tv in [Float64, BigFloat, Int8]
        AT = AbstractSparseTensor{Tv}
        @test indtype(AT) == Integer
        for T in subtypes(AT)
            @test indtype(T) == Integer
        end
    end
    @testset "T::Type{<:AbstractSparseTensor{$Tv,$Ti}}" for Tv in [Float64, BigFloat, Int8], Ti in [Int, UInt8]
        AT = AbstractSparseTensor{Tv,Ti}
        @test indtype(AT) == Ti
        for T in subtypes(AT)
            @test indtype(T) == Ti
        end
    end
    @testset "T::Type{<:AbstractSparseTensor{$Tv,$Ti,$N}}" for Tv in [Float64, BigFloat, Int8], Ti in [Int, UInt8], N in 1:3
        AT = AbstractSparseTensor{Tv,Ti,N}
        @test indtype(AT) == Ti
        for T in subtypes(AT)
            @test indtype(T) == Ti
        end
    end

    # indtype(A::AbstractSparseTensor)
    @testset "A::AbstractSparseTensor{$Tv, $Ti, $N}" for Tv in [Float64, BigFloat, Int8], Ti in [Int, UInt8], N in 1:3
        dims = (5, 3, 2)[1:N]
        inds = (Ti[2, 1, 4], Ti[1, 3, 2], Ti[1, 2, 1])[1:N]
        vals = Tv[1, 0, 10]

        @test indtype(SparseTensorCOO(dims, tuple.(inds...), vals)) == Ti
        @test indtype(SparseTensorDOK(dims, Dict(tuple.(inds...) .=> vals))) == Ti
    end
end
