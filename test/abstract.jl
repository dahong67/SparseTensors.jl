## AbstractSparseTensor type and functions/methods

@testset "indtype" begin
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
