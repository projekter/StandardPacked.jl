using Test
using StandardPacked
using LinearAlgebra, SparseArrays
using Base: _realtype
using StaticArrays

setprecision(8)

function eqapprox(a, b, sca::Symbol; kwargs...)
    if sca === :S
        return LinearAlgebra.isapprox(a, b; kwargs...)
    else
        return a == b
    end
end

@testset "Element type $elty" for elty in (Float32, Float64, BigFloat), tri in (:U, :L), sca in (Symbol(), :S) begin
    refmatrix = elty[0.9555114416827896 0.14061115800771273 0.660907232019832 0.7388163447790721 0.15025179696269242 0.9847133877489483 0.3610824315069938
        0.14061115800771273 0.5391622763612515 0.21722146893315186 0.8336423569126538 0.41116243025455734 0.5343466059451054 0.43066682904413156
        0.660907232019832 0.21722146893315186 0.0019306390378336369 0.22609164615603128 0.35352758268808504 0.644850818020348 0.8100978477428766
        0.7388163447790721 0.8336423569126538 0.22609164615603128 0.5535820925972024 0.40253776414772224 0.2036884435817956 0.26796888784023765
        0.15025179696269242 0.41116243025455734 0.35352758268808504 0.40253776414772224 0.7741030577273389 0.33109536531255035 0.27758928139887895
        0.9847133877489483 0.5343466059451054 0.644850818020348 0.2036884435817956 0.33109536531255035 0.14756643458657948 0.09214560319184117
        0.3610824315069938 0.43066682904413156 0.8100978477428766 0.26796888784023765 0.27758928139887895 0.09214560319184117 0.45340576959493983]
    n = 7
    fmt = Symbol(tri, sca)
    @test packedsize(refmatrix) == 28
    @testset "Format = $fmt" begin
        if fmt === :U
            data = elty[0.9555114416827896,0.14061115800771273,0.5391622763612515,0.660907232019832,0.21722146893315186,
                0.0019306390378336369,0.7388163447790721,0.8336423569126538,0.22609164615603128,0.5535820925972024,
                0.15025179696269242,0.41116243025455734,0.35352758268808504,0.40253776414772224,0.7741030577273389,
                0.9847133877489483,0.5343466059451054,0.644850818020348,0.2036884435817956,0.33109536531255035,
                0.14756643458657948,0.3610824315069938,0.43066682904413156,0.8100978477428766,0.26796888784023765,
                0.27758928139887895,0.09214560319184117,0.45340576959493983]
        elseif fmt === :L
            data = elty[0.9555114416827896,0.14061115800771273,0.660907232019832,0.7388163447790721,0.15025179696269242,
                0.9847133877489483,0.3610824315069938,0.5391622763612515,0.21722146893315186,0.8336423569126538,
                0.41116243025455734,0.5343466059451054,0.43066682904413156,0.0019306390378336369,0.22609164615603128,
                0.35352758268808504,0.644850818020348,0.8100978477428766,0.5535820925972024,0.40253776414772224,
                0.2036884435817956,0.26796888784023765,0.7741030577273389,0.33109536531255035,0.27758928139887895,
                0.14756643458657948,0.09214560319184117,0.45340576959493983]
        elseif fmt === :US
            data = elty[0.9555114416827896,0.19885420667549358,0.5391622763612515,0.9346639709929084,0.3071975474038693,
                0.0019306390378336369,1.0448440948894804,1.1789483273145474,0.3197418723331183,0.5535820925972024,
                0.21248812903556824,0.5814714852042768,0.4999635021104657,0.569274365425051,0.7741030577273389,
                1.392595028004919,0.7556802171356001,0.9119567725517609,0.28805895941204235,0.4682395560638831,
                0.14756643458657948,0.5106476717718449,0.6090548704984261,1.1456513631272307,0.37896523547769884,
                0.3925705265236962,0.13031356174695136,0.45340576959493983]
        elseif fmt === :LS
            data = elty[0.9555114416827896,0.19885420667549358,0.9346639709929084,1.0448440948894804,0.21248812903556824,
                1.392595028004919,0.5106476717718449,0.5391622763612515,0.3071975474038693,1.1789483273145474,
                0.5814714852042768,0.7556802171356001,0.6090548704984261,0.0019306390378336369,0.3197418723331183,
                0.4999635021104657,0.9119567725517609,1.1456513631272307,0.5535820925972024,0.569274365425051,
                0.28805895941204235,0.37896523547769884,0.7741030577273389,0.4682395560638831,0.3925705265236962,
                0.14756643458657948,0.13031356174695136,0.45340576959493983]
        end
        pm = SPMatrix(n, data, fmt)
        @testset "Elementary properties" begin
            @test size(pm) == (n, n)
            @test LinearAlgebra.checksquare(pm) == packedside(pm) == packedside(vec(pm)) == n
            @test eltype(pm) === elty
            @test packed_isupper(pm) == (tri === :U)
            @test packed_islower(pm) == (tri === :L)
            @test packed_isscaled(pm) == (sca === :S)
            @test issymmetric(pm)
            @test ishermitian(pm)
            @test transpose(pm) == pm
        end
        @testset "Conversion and element access" begin
            @test collect(pm) == data
            @test [x for x in pm] == data
            @test eqapprox(Matrix(pm), refmatrix, sca, rtol=2eps(elty))
            if sca === :S
                @test vec(lmul_offdiags!(sqrt(elty(2)), SPMatrix(Symmetric(refmatrix, tri)))) ≈ vec(pm) rtol=2eps(elty)
            else
                @test SPMatrix(Symmetric(refmatrix, tri)) == pm
            end
            @test eqapprox([pm[i, j] for i in 1:n, j in 1:n], refmatrix, sca, rtol=2eps(elty))
            fill!(pm, zero(elty))
            for i in 1:n, j in 1:n
                pm[i, j] = refmatrix[i, j]
            end
            @test eqapprox(Matrix(pm), refmatrix, sca, rtol=2eps(elty))
            fill!(pm, zero(elty))
            for j in 1:n, i in 1:n
                pm[i, j] = refmatrix[i, j]
            end
            @test eqapprox(Matrix(pm), refmatrix, sca, rtol=2eps(elty))
            fill!(pm, zero(elty))
            @test copyto!(pm, refmatrix) === pm
            @test eqapprox(Matrix(pm), refmatrix, sca, rtol=2eps(elty))
            sca === :S && copy!(pm, refmatrix)
            @test_throws ErrorException SPMatrix(pm, tri === :U ? :L : :U)
            @test diag(pm) == diag(refmatrix)
            @test tr(pm) == tr(refmatrix)
        end
        local pmc, otherscaled
        @testset "Scaling and unscaling" begin
            pmc = copy(pm)
            @test pmc == pm
            if sca === :S
                @test packed_scale!(pmc) === pmc && pmc == pm
                otherscaled = packed_unscale!(pmc)
                @test !packed_isscaled(otherscaled)
                @test Matrix(otherscaled) ≈ refmatrix rtol=2eps(elty)
                @test packed_scale!(otherscaled) ≈ pmc rtol=4eps(elty)
            else
                @test packed_unscale!(pmc) === pmc && pmc == pm
                otherscaled = packed_scale!(pmc)
                @test packed_isscaled(otherscaled)
                @test Matrix(otherscaled) ≈ refmatrix rtol=2eps(elty)
                @test packed_unscale!(otherscaled) ≈ pmc rtol=4eps(elty)
            end
        end
        @testset "Functions on the vector and broadcasting" begin
            fill!(pmc, 5.)
            if sca === :S
                @test all(x -> x == 5 || x == 5 / sqrt(2), pmc)
            else
                @test all(x -> x == 5., pmc)
            end
            @test vec(2 .* pm) == 2data
            @test vec(pm .+ pm) == data .+ data
            @test vec(pm .+ data) == data .+ data
            pmc .= pm .+ 3 .* pm
            @test vec(pmc) == 4data
            otherscaled = sca === :S ? packed_unscale!(pmc) : packed_scale!(pmc)
            @test pm .+ otherscaled ≈ 5pm
        end

        if elty <: LinearAlgebra.BlasFloat # we need BLAS support for these tests
            @testset "BLAS functions: spmv and spr" begin
                if sca !== :S # not implemented for scaled matrices
                    tmpout = Vector{elty}(undef, n)
                    @test mul!(tmpout, pm, elty.(collect(1:n)), true, false) === tmpout
                    @test tmpout ≈ elty[15.361837164730108,13.481729135432627,13.312936427386411,10.409306064572332,
                        11.443522912431101,7.988607484452503,9.839245597494859] rtol=2eps(elty)
                end
                copyto!(pmc, pm)
                if sca === :S
                    @test !packed_isscaled(spr!(elty(4), elty.(collect(5:11)), pmc))
                else
                    @test spr!(elty(4), elty.(collect(5:11)), pmc) === pmc
                end
                if tri === :U
                    @test vec(pmc) ≈ elty[100.95551144168279,120.14061115800772,144.53916227636125,140.66090723201984,
                        168.21722146893316,196.00193063903782,160.73881634477908,192.83364235691266,224.22609164615602,
                        256.5535820925972,180.1502517969627,216.41116243025456,252.3535275826881,288.4025377641477,
                        324.77410305772736,200.98471338774894,240.5343466059451,280.64485081802036,320.20368844358177,
                        360.33109536531254,400.14756643458657,220.361082431507,264.4306668290441,308.8100978477429,
                        352.26796888784025,396.2775892813989,440.09214560319185,484.45340576959495] rtol=2eps(elty)
                elseif tri === :L
                    @test vec(pmc) ≈ elty[100.95551144168279,120.14061115800772,140.66090723201984,160.73881634477908,
                        180.1502517969627,200.98471338774894,220.361082431507,144.53916227636125,168.21722146893316,
                        192.83364235691266,216.41116243025456,240.5343466059451,264.4306668290441,196.00193063903782,
                        224.22609164615602,252.3535275826881,280.64485081802036,308.8100978477429,256.5535820925972,
                        288.4025377641477,320.20368844358177,352.26796888784025,324.77410305772736,360.33109536531254,
                        396.2775892813989,400.14756643458657,440.09214560319185,484.45340576959495] rtol=2eps(elty)
                end
            end
        end

        @testset "Scalar product and norm" begin
            pms = SPMatrix(n, sparsevec(data), fmt)
            @test dot(pm, pm) ≈ 12.788602219279593 rtol=2eps(elty)
            @test dot(pm, pms) ≈ 12.788602219279593 rtol=2eps(elty)
            @test dot(pms, pms) ≈ 12.788602219279593 rtol=2eps(elty)
            for p in (pm, pms)
                @test norm(pm) ≈ sqrt(12.788602219279593) rtol=2eps(elty)
                @test norm(pm, 1) ≈ 21.571292275978372 rtol=2eps(elty)
                @test eqapprox(norm(pm, Inf), elty(0.9847133877489483), sca, rtol=2eps(elty))
                @test norm(pm, 0) == 49
                @test eqapprox(norm(pm, -Inf), elty(0.0019306390378336369), sca, rtol=2eps(elty))
                @test norm(pm, 3.) ≈ 2.0766959423281186 rtol=2eps(elty)
            end
            fill!(pmc, zero(elty))
            pmc[diagind(pmc)] .= elty.(3:9)
            pmcs = SPMatrix(n, SparseVector{elty,Int}(packedsize(n), Int[], elty[]), fmt)
            pmcs[diagind(pmcs)] .= elty.(3:9)
            @test dot(pm, pmc) ≈ 19.034233988404225 rtol=2eps(elty)
            @test dot(pm, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
            @test dot(pms, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
            for p in (pmc, pmcs)
                @test norm(p) ≈ 16.73320053068151 rtol=2eps(elty)
                @test norm(p, 1) ≈ 42 rtol=2eps(elty)
                @test norm(p, Inf) == 9
                @test norm(p, 0) == n
                @test norm(p, -Inf) == 0
                @test norm(p, 3.) ≈ 12.632719195312756 rtol=2eps(elty)
            end
        end

        pmc .= pm
        @test pmc == pm
        if elty <: LinearAlgebra.BlasFloat # we need LAPACK support for these tests
            @testset "Eigendecomposition" begin
                es = eigen!(pmc)
                @test es.values ≈ elty[-0.9942609100484255,-0.6811215090551395,0.005159417098787866,0.40031013318133934,
                    0.5934629480373085,0.952760032261558,3.1489516001124938]
                for (checkvec, refvec) in zip(eachcol(es.vectors), eachcol(elty[0.2889606455922179 -0.44020703951530277 0.24348028102926525 0.1204919634041064 -0.19273557783801398 -0.5975107397878534 -0.5047300863220398
                    0.3560852991851457 -0.25356244792120586 -0.534776235898921 -0.29478952295818195 -0.2805041045202071 0.47675402154293717 -0.36061878718426976
                    0.4662626441009365 0.6660011327317795 -0.08857321145700488 -0.07131124215247894 0.4224637849418201 -0.15489416472931736 -0.3516294626681506
                    -0.26326864417213147 0.39734156630104617 0.5033974466315662 -0.20617971628138693 -0.5077157386268769 0.24447010053694065 -0.39918583980167427
                    0.030009939767552042 -0.13063176272830587 0.1993775338626156 0.721463578470526 0.2696244156761315 0.5069046053197481 -0.30351943198079384
                    -0.612091846007436 0.15776570792564273 -0.5697957286958822 0.255469158030155 -0.05649395724288717 -0.25743043518462644 -0.3755794112367116
                    -0.35698284031768523 -0.31389702487282756 0.17046686996450022 -0.5149866692272487 0.60994313735979 0.09330583370485736 -0.3146825016453474]))
                    @test ≈(checkvec, refvec, atol=50eps(elty)) || ≈(checkvec, -refvec, atol=50eps(elty))
                end
                @test eigvals(pm) ≈ es.values
                @test vec(pm) == data
                es2 = eigen(pm, elty(-.7), elty(1.))
                @test es2.values ≈ @view(es.values[2:6])
                for (checkvec, refvec) in zip(eachcol(es2.vectors), eachcol(@view(es.vectors[:, 2:6])))
                    @test ≈(checkvec, refvec, atol=16eps(elty)) || ≈(checkvec, -refvec, atol=16eps(elty))
                end
                es3 = eigen(pm, 3:5)
                @test es3.values ≈ @view(es.values[3:5]) rtol=4eps(elty)
                for (checkvec, refvec) in zip(eachcol(es3.vectors), eachcol(@view(es.vectors[:, 3:5])))
                    @test ≈(checkvec, refvec, atol=8eps(elty)) || ≈(checkvec, -refvec, atol=8eps(elty))
                end
                @test eigvals(pm, elty(-.7), elty(1.)) ≈ @view(es.values[2:6])
                @test eigvals(pm, 3:5) ≈ @view(es.values[3:5])
                @test eigmin(pm) ≈ minimum(es.values)
                @test eigmax(pm) ≈ maximum(es.values)
                @test eigvecs(pm) == es.vectors

                # functions that are not called by the LinearAlgebra interface
                @test spev!('N', copy(pm)) ≈ es.values
                let newev=spev!('V', copy(pm))
                    @test newev[1] ≈ es.values
                    @test newev[2] ≈ es.vectors
                end
                @test spevx!('N', copy(pm)) ≈ es.values
                @test spevx!('N', StandardPacked.packed_ulchar(pm), vec(packed_unscale!(copy(pm)))) ≈ es.values
                let newev=spevx!('V', copy(pm))
                    @test newev[1] ≈ es.values
                    @test newev[2] ≈ es.vectors
                end
                let newev=spevx!('V', StandardPacked.packed_ulchar(pm), vec(packed_unscale!(copy(pm))))
                    @test newev[1] ≈ es.values
                    @test newev[2] ≈ es.vectors
                end

                # check pre-allocated routines (eigenvalues only)
                # spevd
                @test_throws DimensionMismatch eigvals!(pm, Vector{elty}(undef, n +1))
                @test_throws DimensionMismatch eigvals!(pm, Vector{elty}(undef, n -1))
                @test_throws ArgumentError eigvals!(pm, missing, Vector{elty}(undef, 2n -1))
                @test_throws ArgumentError eigvals!(pm, missing, missing, Vector{Int}(undef, 0))
                @test eigvals(pm, Vector{elty}(undef, n), Vector{elty}(undef, 2n), Vector{Int}(undef, 1)) ≈ es.values
                # spev
                @test_throws DimensionMismatch spev!('N', pm, Vector{elty}(undef, n +1))
                @test_throws DimensionMismatch spev!('N', pm, Vector{elty}(undef, n -1))
                @test_throws ArgumentError spev!('N', pm, missing, Vector{elty}(undef, 3n -1))
                @test spev!('N', copy(pm), Vector{elty}(undef, n), Vector{elty}(undef, 3n)) ≈ es.values
                # spevx interval range
                @test_throws ArgumentError eigvals!(pm, elty(-.7), elty(1.), missing, Vector{elty}(undef, 8n -1))
                @test_throws ArgumentError eigvals!(pm, elty(-.7), elty(1.), missing, missing, Vector{Int}(undef, 5n -1))
                evstore = fill(elty(42), n)
                @test eigvals(pm, elty(-.7), elty(1.), evstore, Vector{elty}(undef, 8n), Vector{Int}(undef, 5n)) ≈ @view(es.values[2:6])
                @test all(x -> x == elty(42.), @view(evstore[6:end]))
                # spevx index range
                @test_throws DimensionMismatch eigvals!(pm, 3:5, Vector{elty}(undef, 3))
                # spevx total
                @test_throws DimensionMismatch spevx!('N', 'A', pm, nothing, nothing, nothing, nothing, elty(-1),
                    Vector{elty}(undef, n -1))

                # check pre-allocated routines (including eigenvectors)
                # spevd
                @test_throws DimensionMismatch eigen!(pm, Vector{elty}(undef, n +1))
                @test_throws DimensionMismatch eigen!(pm, Vector{elty}(undef, n -1))
                @test_throws DimensionMismatch eigen!(pm, missing, Matrix{elty}(undef, 1, n))
                @test_throws DimensionMismatch eigen!(pm, missing, Matrix{elty}(undef, n -1, n -1))
                @test_throws DimensionMismatch eigen!(pm, missing, Matrix{elty}(undef, n +1, n +1))
                @test_throws ArgumentError eigen!(pm, missing, missing, Vector{elty}(undef, n^2 + 6n))
                @test_throws ArgumentError eigen!(pm, missing, missing, missing, Vector{Int}(undef, 2 + 5n))
                @test eigen(pm, Vector{elty}(undef, n), Matrix{elty}(undef, n, n), Vector{elty}(undef, n^2 + 6n +1),
                    Vector{Int}(undef, 3 + 5n)) == es
                # spev
                @test_throws DimensionMismatch spev!('V', pm, Vector{elty}(undef, n +1))
                @test_throws DimensionMismatch spev!('V', pm, Vector{elty}(undef, n -1))
                @test_throws DimensionMismatch spev!('V', pm, missing, Matrix{elty}(undef, 1, n))
                @test_throws DimensionMismatch spev!('V', pm, missing, Matrix{elty}(undef, n -1, n -1))
                @test_throws DimensionMismatch spev!('V', pm, missing, Matrix{elty}(undef, n +1, n +1))
                @test_throws ArgumentError spev!('V', pm, missing, missing, Vector{elty}(undef, 3n -1))
                let newev=spev!('V', copy(pm), Vector{elty}(undef, n), Matrix{elty}(undef, n, n), Vector{elty}(undef, 3n))
                    @test newev[1] ≈ es.values
                    @test newev[2] ≈ es.vectors
                end
                # spevx interval range
                @test_throws DimensionMismatch eigen!(pm, elty(-.7), elty(1.), missing, Matrix{elty}(undef, 1, n))
                @test_throws DimensionMismatch eigen!(pm, elty(-.7), elty(1.), missing, Matrix{elty}(undef, n +1, n))
                @test_throws ArgumentError eigen!(pm, elty(-.7), elty(1.), missing, missing, Vector{elty}(undef, 8n -1))
                @test_throws ArgumentError eigen!(pm, elty(-.7), elty(1.), missing, missing, missing, Vector{Int}(undef, 5n -1))
                @test_throws DimensionMismatch eigen!(pm, elty(-.7), elty(1.), missing, missing, missing, missing,
                    Vector{Int}(undef, n -1))
                evstore = fill(elty(42), n)
                evecstore = fill(elty(42), n, n)
                @test eigen(pm, elty(-.7), elty(1.), evstore, evecstore, Vector{elty}(undef, 8n), Vector{Int}(undef, 5n),
                    Vector{Int}(undef, n)) == es2
                @test all(x -> x == elty(42.), @view(evstore[6:end]))
                @test all(x -> x == elty(42.), @view(evecstore[:, 6:end]))
                # spevx index range
                @test_throws DimensionMismatch eigen!(pm, 3:5, Vector{elty}(undef, 3))
                @test_throws DimensionMismatch eigen!(pm, 3:5, missing, Matrix{elty}(undef, n, 2))
                # spevx total
                @test_throws DimensionMismatch spevx!('V', 'A', pm, nothing, nothing, nothing, nothing, elty(-1),
                    Vector{elty}(undef, n -1))
                @test_throws DimensionMismatch spevx!('V', 'A', pm, nothing, nothing, nothing, nothing, elty(-1),
                    missing, Matrix{elty}(undef, n, n -1))
            end

            @testset "Generalized eigendecomposition" begin
                if fmt === :U
                    bdata = elty[1.5519702373967077, 0.24098484568907816, 2.2605926076675775, -1.7569322370503215,
                        -0.5039320101981892, 3.7278698348111394, 0.3483700140851673, 2.0181906496649784, -1.226263452351277,
                        2.4287428457369566, -0.34260565607043714, -2.3248630684215614, 0.9993405545447425, -2.2793613104638037,
                        2.8088828475702017, -0.13665367477077597, -0.5708164643148312, 1.5673420879502815, -0.7434324369522352,
                        0.9457774768648077, 2.7118723016325204, 1.2385104373180598, 0.8060574708934514, -2.0133651895165974,
                        1.65105901512575, -0.8386047121950104, 0.40531794375248753, 3.326008695109104]
                elseif fmt === :L
                    bdata = elty[1.5519702373967077, 0.24098484568907816, -1.7569322370503215, 0.3483700140851673,
                        -0.34260565607043714, -0.13665367477077597, 1.2385104373180598, 2.2605926076675775, -0.5039320101981892,
                        2.0181906496649784, -2.3248630684215614, -0.5708164643148312, 0.8060574708934514, 3.7278698348111394,
                        -1.226263452351277, 0.9993405545447425, 1.5673420879502815, -2.0133651895165974, 2.4287428457369566,
                        -2.2793613104638037, -0.7434324369522352, 1.65105901512575, 2.8088828475702017, 0.9457774768648077,
                        -0.8386047121950104, 2.7118723016325204, 0.40531794375248753, 3.326008695109104]
                elseif fmt === :US
                    bdata = elty[1.5519702373967077, 0.34080403709988183, 2.2605926076675775, -2.4846773978070664,
                        -0.7126674833362161, 3.7278698348111394, 0.4926695986433498, 2.8541525882107806, -1.7341984053576296,
                        2.4287428457369566, -0.48451756536054436, -3.287852882022101, 1.4132809656666245, -3.223503678806422,
                        2.8088828475702017, -0.19325748020895345, -0.8072563854598922, 2.2165564376574527, -1.051372235045932,
                        1.3375313347692173, 2.7118723016325204, 1.7515182575958332, 1.1399374073896753, -2.847328357024249,
                        2.334950051469201, -1.18596615745617, 0.5732061331279432, 3.326008695109104]
                elseif fmt === :LS
                    bdata = elty[1.5519702373967077, 0.34080403709988183, -2.4846773978070664, 0.4926695986433498,
                        -0.48451756536054436, -0.19325748020895345, 1.7515182575958332, 2.2605926076675775,
                        -0.7126674833362161, 2.8541525882107806, -3.287852882022101, -0.8072563854598922, 1.1399374073896753,
                        3.7278698348111394, -1.7341984053576296, 1.4132809656666245, 2.2165564376574527, -2.847328357024249,
                        2.4287428457369566, -3.223503678806422, -1.051372235045932, 2.334950051469201, 2.8088828475702017,
                        1.3375313347692173, -1.18596615745617, 2.7118723016325204, 0.5732061331279432, 3.326008695109104]
                end
                pmb = SPMatrix(n, bdata, fmt)
                es = eigen(pm, pmb)
                @test es.values ≈ elty[-1.8318388366067977, -0.3922353401226242, 0.0029830752157166502, 0.06152008931674012,
                    0.9448050424793925, 3.6505581997040495, 12.606653507880447]
                for (checkvec, refvec) in zip(eachcol(es.vectors), eachcol(elty[-0.3037306291917914 0.1849449315681271 -0.1820511515991851 -0.08464157332217992 -0.08301101851587915 -1.3847290706343123 -0.8290787765040575
                    -1.2766535078335355 -0.5152769472886553 0.4064930607080493 0.2785978865385685 0.21973769919298894 1.5751940654040593 -0.5056022332777412
                    -0.22437585580079472 0.6515226299045409 0.06352595729023604 0.00017423361080127207 0.6863161097088751 -1.0229822048040997 -0.05527377167291614
                    1.4759281213703794 0.568521872715543 -0.38878679240664116 0.042063505064296375 -0.5476305946062237 -0.7314494681859702 -2.001420698419801
                    -0.10283912802833545 -0.020725355270067417 -0.1468435733677217 -0.32235501891586565 0.07216131811395694 0.8868569748143307 -1.8406383341740356
                    0.5164654845363702 -0.6164510876007171 0.435444949138487 0.04728590028705108 -0.5256217439323876 0.23843087778223537 -0.22235592814603297
                    -0.43262674989138933 -0.08638368373728088 -0.1286787160851502 0.055142325493374104 1.0388690404022434 0.02562085002921159 0.9254691349230983]))
                    @test ≈(checkvec, refvec, atol=75eps(elty)) || ≈(checkvec, -refvec, atol=75eps(elty))
                end
                @test eigvals(pm, pmb) ≈ es.values
                @test vec(pm) == data && vec(pmb) == bdata
                es2 = spgvx!(1, 'V', 'V', copy(pm), copy(pmb), elty(-.7), elty(1.), nothing, nothing, elty(-1))
                @test es2[1] ≈ @view(es.values[2:5])
                for (checkvec, refvec) in zip(eachcol(es2[2]), eachcol(@view(es.vectors[:, 2:5])))
                    @test ≈(checkvec, refvec, atol=16eps(elty)) || ≈(checkvec, -refvec, atol=16eps(elty))
                end
                es3 = spgvx!(1, 'V', 'I', copy(pm), copy(pmb), nothing, nothing, 3, 5, elty(-1))
                @test es3[1] ≈ @view(es.values[3:5]) rtol=16eps(elty)
                for (checkvec, refvec) in zip(eachcol(es3[2]), eachcol(@view(es.vectors[:, 3:5])))
                    @test ≈(checkvec, refvec, atol=16eps(elty)) || ≈(checkvec, -refvec, atol=16eps(elty))
                end

                # functions that are not called by the LinearAlgebra interface
                @test spgv!(1, 'N', copy(pm), copy(pmb))[1] ≈ es.values
                let newev=spgv!(1, 'V', copy(pm), copy(pmb))
                    @test newev[1] ≈ es.values
                    @test newev[2] ≈ es.vectors
                end
                @test spgvx!(1, 'N', copy(pm), copy(pmb))[1] ≈ es.values
                @test spgvx!(1, 'N', StandardPacked.packed_ulchar(pm), vec(packed_unscale!(copy(pm))),
                    vec(packed_unscale!(copy(pmb))))[1] ≈ es.values
                let newev=spgvx!(1, 'V', copy(pm), copy(pmb))
                    @test newev[1] ≈ es.values
                    @test newev[2] ≈ es.vectors
                end
                let newev=spgvx!(1, 'V', StandardPacked.packed_ulchar(pm), vec(packed_unscale!(copy(pm))),
                        vec(packed_unscale!(copy(pmb))))
                    @test newev[1] ≈ es.values
                    @test newev[2] ≈ es.vectors
                end

                # check pre-allocated routines (eigenvalues only)
                # spgvd
                @test_throws DimensionMismatch spgvd!(1, 'N', pm, pmb, Vector{elty}(undef, n +1))
                @test_throws DimensionMismatch spgvd!(1, 'N', pm, pmb, Vector{elty}(undef, n -1))
                @test_throws ArgumentError spgvd!(1, 'N', pm, pmb, missing, Vector{elty}(undef, 2n -1))
                @test_throws ArgumentError spgvd!(1, 'N', pm, pmb, missing, missing, Vector{Int}(undef, 0))
                @test spgvd!(1, 'N', copy(pm), copy(pmb), Vector{elty}(undef, n), Vector{elty}(undef, 2n),
                    Vector{Int}(undef, 1))[1] ≈ es.values
                # spgv
                @test_throws DimensionMismatch spgv!(1, 'N', pm, pmb, Vector{elty}(undef, n +1))
                @test_throws DimensionMismatch spgv!(1, 'N', pm, pmb, Vector{elty}(undef, n -1))
                @test_throws ArgumentError spgv!(1, 'N', pm, pmb, missing, Vector{elty}(undef, 3n -1))
                @test spgv!(1, 'N', copy(pm), copy(pmb), Vector{elty}(undef, n), Vector{elty}(undef, 3n))[1] ≈ es.values
                # spgvx interval range
                @test_throws ArgumentError spgvx!(1, 'N', 'V', pm, pmb, elty(-.7), elty(1.), nothing, nothing, elty(-1),
                    missing, Vector{elty}(undef, 8n -1))
                @test_throws ArgumentError spgvx!(1, 'N', 'V', pm, pmb, elty(-.7), elty(1.), nothing, nothing, elty(-1),
                    missing, missing, Vector{Int}(undef, 5n -1))
                evstore = fill(elty(42), n)
                @test spgvx!(1, 'N', 'V', copy(pm), copy(pmb), elty(-.7), elty(1.), nothing, nothing, elty(-1), evstore,
                    Vector{elty}(undef, 8n), Vector{Int}(undef, 5n))[1] ≈ @view(es.values[2:5])
                @test all(x -> x == elty(42.), @view(evstore[5:end]))
                # spgvx index range
                @test_throws DimensionMismatch spgvx!(1, 'N', 'I', pm, pmb, nothing, nothing, 3, 5, elty(-1),
                    Vector{elty}(undef, 2))
                # spevx total
                @test_throws DimensionMismatch spgvx!(1, 'N', 'A', pm, pmb, nothing, nothing, nothing, nothing, elty(-1),
                    Vector{elty}(undef, n -1))

                # check pre-allocated routines (including eigenvectors)
                # spgvd
                @test_throws DimensionMismatch spgvd!(1, 'V', pm, pmb, Vector{elty}(undef, n +1))
                @test_throws DimensionMismatch spgvd!(1, 'V', pm, pmb, Vector{elty}(undef, n -1))
                @test_throws DimensionMismatch spgvd!(1, 'V', pm, pmb, missing, Matrix{elty}(undef, 1, n))
                @test_throws DimensionMismatch spgvd!(1, 'V', pm, pmb, missing, Matrix{elty}(undef, n -1, n -1))
                @test_throws DimensionMismatch spgvd!(1, 'V', pm, pmb, missing, Matrix{elty}(undef, n +1, n +1))
                @test_throws ArgumentError spgvd!(1, 'V', pm, pmb, missing, missing, Vector{elty}(undef, 2n^2 + 6n))
                @test_throws ArgumentError spgvd!(1, 'V', pm, pmb, missing, missing, missing, Vector{Int}(undef, 2 + 5n))
                let newev=spgvd!(1, 'V', copy(pm), copy(pmb), Vector{elty}(undef, n), Matrix{elty}(undef, n, n),
                    Vector{elty}(undef, 2n^2 + 6n +1), Vector{Int}(undef, 3 + 5n))
                    @test newev[1] == es.values
                    @test newev[2] == es.vectors
                end
                # spgv
                @test_throws DimensionMismatch spgv!(1, 'V', pm, pmb, Vector{elty}(undef, n +1))
                @test_throws DimensionMismatch spgv!(1, 'V', pm, pmb, Vector{elty}(undef, n -1))
                @test_throws DimensionMismatch spgv!(1, 'V', pm, pmb, missing, Matrix{elty}(undef, 1, n))
                @test_throws DimensionMismatch spgv!(1, 'V', pm, pmb, missing, Matrix{elty}(undef, n -1, n -1))
                @test_throws DimensionMismatch spgv!(1, 'V', pm, pmb, missing, Matrix{elty}(undef, n +1, n +1))
                @test_throws ArgumentError spgv!(1, 'V', pm, pmb, missing, missing, Vector{elty}(undef, 3n -1))
                let newev=spgv!(1, 'V', copy(pm), copy(pmb), Vector{elty}(undef, n), Matrix{elty}(undef, n, n),
                    Vector{elty}(undef, 3n))
                    @test newev[1] ≈ es.values
                    @test newev[2] ≈ es.vectors
                end
                # spgvx interval range
                @test_throws DimensionMismatch spgvx!(1, 'V', 'V', pm, pmb, elty(-.7), elty(1.), nothing, nothing, elty(-1),
                    missing, Matrix{elty}(undef, 1, n))
                @test_throws DimensionMismatch spgvx!(1, 'V', 'V', pm, pmb, elty(-.7), elty(1.), nothing, nothing, elty(-1),
                    missing, Matrix{elty}(undef, n +1, n))
                @test_throws ArgumentError spgvx!(1, 'V', 'V', pm, pmb, elty(-.7), elty(1.), nothing, nothing, elty(-1),
                    missing, missing, Vector{elty}(undef, 8n -1))
                @test_throws ArgumentError spgvx!(1, 'V', 'V', pm, pmb, elty(-.7), elty(1.), nothing, nothing, elty(-1),
                    missing, missing, missing, Vector{Int}(undef, 5n -1))
                @test_throws DimensionMismatch spgvx!(1, 'V', 'V', pm, pmb, elty(-.7), elty(1.), nothing, nothing, elty(-1),
                    missing, missing, missing, missing, Vector{Int}(undef, n -1))
                evstore = fill(elty(42), n)
                evecstore = fill(elty(42), n, n)
                @test spgvx!(1, 'V', 'V', copy(pm), copy(pmb), elty(-.7), elty(1.), nothing, nothing, elty(-1), evstore,
                    evecstore, Vector{elty}(undef, 8n), Vector{Int}(undef, 5n), Vector{Int}(undef, n)) == es2
                @test all(x -> x == elty(42.), @view(evstore[6:end]))
                @test all(x -> x == elty(42.), @view(evecstore[:, 6:end]))
                # spegx index range
                @test_throws DimensionMismatch spgvx!(1, 'V', 'I', pm, pmb, nothing, nothing, 3, 5, elty(-1),
                    Vector{elty}(undef, 2))
                @test_throws DimensionMismatch spgvx!(1, 'V', 'I', pm, pmb, nothing, nothing, 3, 5, elty(-1), missing,
                    Matrix{elty}(undef, n, 2))
                # spgvx total
                @test_throws DimensionMismatch spgvx!(1, 'V', 'A', pm, pmb, nothing, nothing, nothing, nothing, elty(-1),
                    Vector{elty}(undef, n -1))
                @test_throws DimensionMismatch spgvx!(1, 'V', 'A', pm, pmb, nothing, nothing, nothing, nothing, elty(-1),
                    missing, Matrix{elty}(undef, n, n -1))
            end

            rhs = elty[1.0 8.0; 2.0 9.0; 3.0 10.0; 4.0 11.0; 5.0 12.0; 6.0 13.0; 7.0 14.0]
            @testset "Cholesky decomposition" begin
                copyto!(pmc, pm)
                @test !LinearAlgebra.isposdef(pmc)
                @test_throws PosDefException cholesky(pmc)
                @test pmc == pm
                @test LinearAlgebra.isposdef(pmc, one(elty))
                chol = cholesky!(pmc, shift=one(elty))
                @test chol.U ≈ elty[1.3983960246234932 0.10055174323423234 0.4726180712633075 0.5283312679453536 0.10744581242866912 0.7041734747594656 0.25821185497449645
                    0. 1.2365482696982792 0.13723596736343546 0.6312068408830044 0.32377107818793416 0.37486667271940743 0.3272845766073084
                    0. 0. 0.8716243956057028 -0.12646679080935472 0.29635897553532703 0.298982588596196 0.737871778986569
                    0. 0. 0. 0.9273792313575595 0.19289178031430956 -0.3959064949144887 0.019710925942854545
                    0. 0. 0. 0. 1.2380205421376222 0.09840290872367227 -0.06348584599245834
                    0. 0. 0. 0. 0. 0.5053369729908973 -0.8290079246701668
                    0. 0. 0. 0. 0. 0. 0.20854097637370994]
                if fmt === :U
                    @test inv(chol) ≈ SPMatrix(n, elty[15.493628236589293, 18.055791818905497, 22.83884746037381,
                        22.56819523083977, 28.380477703283688, 39.43723994306926, -13.664618497320284, -16.691276904083473,
                        -20.71296414574622, 13.085254087036088, 1.6509049892926726, 1.8105792940720054, 2.1679624382488236,
                        -1.5390396304367675, 0.8211060409222076, -31.02109455183003, -37.87395046635378, -49.32438789295014,
                        27.97393922067559, -3.29559139381791, 65.7991061056096, -17.607661895505345, -21.93913378183722,
                        -29.465880029842584, 15.993513682029798, -1.819156241457544, 37.72200796466474, 22.99414125333774]) rtol=150eps(elty)
                elseif fmt === :L
                    @test inv(chol) ≈ SPMatrix(n, elty[15.493628236589293, 18.05579181890549, 22.568195230839763,
                        -13.664618497320282, 1.6509049892926717, -31.02109455183003, -17.60766189550534, 22.83884746037381,
                        28.380477703283688, -16.691276904083473, 1.8105792940720045, -37.87395046635378, -21.93913378183722,
                        39.43723994306926, -20.71296414574622, 2.1679624382488223, -49.32438789295014, -29.465880029842584,
                        13.085254087036088, -1.5390396304367668, 27.97393922067559, 15.993513682029798, 0.8211060409222076,
                        -3.295591393817912, -1.819156241457545, 65.7991061056096, 37.72200796466474, 22.99414125333774], :L) rtol=50eps(elty)
                end
                @test rhs ≈ (pm + I) * (chol \ rhs)
            end

            @testset "Bunch-Kaufman decomposition" begin
                bk = bunchkaufman(pm)
                if fmt === :U
                    let P=bk.P, U=bk.U, D=bk.D
                        @test P' * U * D * U' * P ≈ Matrix(pm)
                    end
                    @test inv(bk) ≈ SPMatrix(n, elty[11.676181149869125, -25.74304267079464, 55.837437314564454,
                        -3.890135969716243, 9.077199222177283, 1.0285759205945262, 24.102856515354063, -51.37534555845253,
                        -9.226894089251159,  49.46819607136643, 9.176107458680693, -21.095279400248227, -3.294216491388214,
                        19.103438949417768, 9.400381615980521, -26.29273501144287, 59.09033629509751, 9.912693866880879,
                        -55.950010085611865, -21.636130300323575, 62.79650629970832, 7.584280750861903, -17.484312579760637,
                        -1.9062117598362571, 16.528078653765096, 5.966940819148872, -19.347475787322477, 6.689363759583235])
                elseif fmt === :L
                    let P=bk.P, L=bk.L, D=bk.D
                        @test P' * L * D * L' * P ≈ Matrix(pm)
                    end
                    @test inv(bk) ≈ SPMatrix(n, elty[11.676181149869125, -25.743042670794633, -3.8901359697162454,
                        24.102856515354063, 9.176107458680692, -26.29273501144288, 7.584280750861906, 55.837437314564454,
                        9.07719922217729, -51.37534555845255, -21.09527940024823, 59.090336295097536, -17.484312579760644,
                        1.0285759205945262, -9.226894089251154, -3.294216491388212, 9.912693866880876, -1.9062117598362565,
                        49.46819607136643, 19.103438949417765, -55.95001008561188, 16.528078653765096, 9.400381615980521,
                        -21.636130300323586, 5.966940819148875, 62.79650629970832, -19.347475787322477, 6.689363759583235], :L)
                end
                @test rhs ≈ pm * (bk \ rhs)
            end
        end
    end
end end

Base.eps(::Type{Complex{R}}) where {R} = 10eps(R)

@testset "Element type $elty" for elty in (ComplexF32, ComplexF64), tri in (:U, :L), sca in (Symbol(), :S) begin
    relty = _realtype(elty)
    refmatrix = elty[.03909956936582404 -0.6653548496259911-0.4034910454602705im 0.04574035368556806+0.19774365201117572im -0.062111947388381994+0.28506946810577816im -0.7564704149996682+0.5038621582694096im 0.6127365820619515+0.4078086300996653im
        -0.6653548496259911+0.4034910454602705im -0.21987747646493983 0.43198514442291525+0.013865715642822352im 0.8042904109643902+0.48443524456616616im -0.4227085640179373+0.020595067313212034im -0.6270868265430396+0.6257143351409404im
        0.04574035368556806-0.19774365201117572im 0.43198514442291525-0.013865715642822352im -0.23335280975427253 -0.07488615770724216-0.9126837555435034im 0.11229843774542969-0.3296891816655916im -0.02289037222879564+0.2065559667686827im
        -0.062111947388381994-0.28506946810577816im 0.8042904109643902-0.48443524456616616im -0.07488615770724216+0.9126837555435034im 0.8551248015983308 0.4973659133644108-0.5138623742474395im 0.7207937158101068+0.6820169266571727im
        -0.7564704149996682-0.5038621582694096im -0.4227085640179373-0.020595067313212034im 0.11229843774542969+0.3296891816655916im 0.4973659133644108+0.5138623742474395im -0.2768539177884062 -0.7553399131583927-0.17575244657065636im
        0.6127365820619515-0.4078086300996653im -0.6270868265430396-0.6257143351409404im -0.02289037222879564-0.2065559667686827im 0.7207937158101068-0.6820169266571727im -0.7553399131583927+0.17575244657065636im -0.020565681831590243]
    n = 6
    fmt = Symbol(tri, sca)
    @test packedsize(refmatrix) == 21
    @testset "Format = $fmt" begin
        if fmt === :U
            data = elty[0.03909956936582404,-0.6653548496259911-0.4034910454602705im,-0.21987747646493983,
                0.04574035368556806+0.19774365201117572im,0.43198514442291525+0.013865715642822352im,-0.23335280975427253,
                -0.062111947388381994+0.28506946810577816im,0.8042904109643902+0.48443524456616616im,
                -0.07488615770724216-0.9126837555435034im,0.8551248015983308,-0.7564704149996682+0.5038621582694096im,
                -0.4227085640179373+0.020595067313212034im,0.11229843774542969-0.3296891816655916im,
                0.4973659133644108-0.5138623742474395im,-0.2768539177884062,0.6127365820619515+0.4078086300996653im,
                -0.6270868265430396+0.6257143351409404im,-0.02289037222879564+0.2065559667686827im,
                0.7207937158101068+0.6820169266571727im,-0.7553399131583927-0.17575244657065636im,-0.020565681831590243]
        elseif fmt === :L
            data = elty[0.03909956936582404,-0.6653548496259911+0.4034910454602705im,0.04574035368556806-0.19774365201117572im,
                -0.062111947388381994-0.28506946810577816im,-0.7564704149996682-0.5038621582694096im,
                0.6127365820619515-0.4078086300996653im,-0.21987747646493983,0.43198514442291525-0.013865715642822352im,
                0.8042904109643902-0.48443524456616616im,-0.4227085640179373-0.020595067313212034im,
                -0.6270868265430396-0.6257143351409404im,-0.23335280975427253,-0.07488615770724216+0.9126837555435034im,
                0.11229843774542969+0.3296891816655916im,-0.02289037222879564-0.2065559667686827im,0.8551248015983308,
                0.4973659133644108+0.5138623742474395im,0.7207937158101068-0.6820169266571727im,-0.2768539177884062,
                -0.7553399131583927+0.17575244657065636im,-0.020565681831590243]
        elseif fmt === :US
            data = elty[0.03909956936582404,-0.940953852131788-0.5706225087860136im,-0.21987747646493983,
                0.06468662852987253+0.2796517545473905im,0.610919249986587+0.019609083114088148im,-0.23335280975427253,
                -0.08783955838205397+0.4031491080136759im,1.137438407272471+0.6850948929569994im,
                -0.10590501986359235-1.290729745247233im,0.8551248015983308,-1.0698107204265344+0.712568697791178im,
                -0.5978001841654227+0.02912582351233128im,0.15881397369289738-0.4662509120791668im,
                0.7033816201420316-0.7267111388539681im,-0.2768539177884062,0.8665403845141467+0.5767284955397395im,
                -0.8868346948826712+0.8848936989275821im,-0.03237187485373125+0.2921142495933574im,
                1.0193562485719514+0.9645175874465902im,-1.0682119493903148-0.248551493560475im,-0.020565681831590243]
        elseif fmt === :LS
            data = elty[0.03909956936582404,-0.940953852131788+0.5706225087860136im,0.06468662852987253-0.2796517545473905im,
                -0.08783955838205397-0.4031491080136759im,-1.0698107204265344-0.712568697791178im,
                0.8665403845141467-0.5767284955397395im,-0.21987747646493983,0.610919249986587-0.019609083114088148im,
                1.137438407272471-0.6850948929569994im,-0.5978001841654227-0.02912582351233128im,
                -0.8868346948826712-0.8848936989275821im,-0.23335280975427253,-0.10590501986359235+1.290729745247233im,
                0.15881397369289738+0.4662509120791668im,-0.03237187485373125-0.2921142495933574im,0.8551248015983308,
                0.7033816201420316+0.7267111388539681im,1.0193562485719514-0.9645175874465902im,-0.2768539177884062,
                -1.0682119493903148+0.248551493560475im,-0.020565681831590243]
        end
        pm = SPMatrix(n, data, fmt)
        @testset "Elementary properties" begin
            @test size(pm) == (n, n)
            @test LinearAlgebra.checksquare(pm) == packedside(pm) == packedside(vec(pm)) == n
            @test eltype(pm) === elty
            @test packed_isupper(pm) == (tri === :U)
            @test packed_islower(pm) == (tri === :L)
            @test packed_isscaled(pm) == (sca === :S)
            @test !issymmetric(pm) # should fall back to a general test
            @test ishermitian(pm)
            @test transpose(pm) != pm
        end
        @testset "Conversion and element access" begin
            @test collect(pm) == data
            @test [x for x in pm] == data
            @test eqapprox(Matrix(pm), refmatrix, sca, rtol=2eps(elty))
            if sca === :S
                @test vec(lmul_offdiags!(sqrt(elty(2)), SPMatrix(Symmetric(refmatrix, tri)))) ≈ vec(pm) rtol=2eps(elty)
            else
                @test SPMatrix(Symmetric(refmatrix, tri)) == pm
            end
            @test eqapprox([pm[i, j] for i in 1:n, j in 1:n], refmatrix, sca, rtol=2eps(elty))
            fill!(pm, zero(elty))
            for i in 1:n, j in 1:n
                pm[i, j] = refmatrix[i, j]
            end
            @test eqapprox(Matrix(pm), refmatrix, sca, rtol=2eps(elty))
            fill!(pm, zero(elty))
            for j in 1:n, i in 1:n
                pm[i, j] = refmatrix[i, j]
            end
            @test eqapprox(Matrix(pm), refmatrix, sca, rtol=2eps(elty))
            fill!(pm, zero(elty))
            @test copyto!(pm, refmatrix) === pm
            @test eqapprox(Matrix(pm), refmatrix, sca, rtol=2eps(elty))
            sca === :S && copy!(pm, refmatrix)
            @test_throws ErrorException SPMatrix(pm, tri === :U ? :L : :U)
            @test diag(pm) == diag(refmatrix)
            @test tr(pm) == tr(refmatrix)
        end
        pmc = copy(pm)

        if elty <: LinearAlgebra.BlasFloat # we need BLAS support for these tests
            @testset "BLAS functions: hpmv and hpr" begin
                if sca !== :S # not implemented for scaled matrices
                    tmpout = Vector{elty}(undef, n)
                    @test mul!(tmpout, pm, elty.(collect(1:n)), true, false) === tmpout
                    @test tmpout ≈ elty[-1.5087694410096137+5.892689309481138im,-2.468056504777488+6.240090518065105im,
                        0.3342575377939869-4.285320213186695im,11.55390146949469+3.006900998098238im,
                        -5.191837644233997+1.4449500696667612im,-1.7270269812515924-4.128210674463004im] rtol=2eps(elty)
                end
                copyto!(pmc, pm)
                if sca === :S
                    @test !packed_isscaled(hpr!(relty(4), elty.(collect(5:10)) .+ elty.(collect(11:16) .* im), pmc))
                else
                    @test hpr!(relty(4), elty.(collect(5:10)) .+ elty.(collect(11:16) .* im), pmc) === pmc
                end
                if tri === :U
                    @test vec(pmc) ≈ elty[584.0390995693658,647.3346451503741+23.59650895453973im,719.780122523535,
                        712.0457403536856+48.19774365201118im,792.4319851444229+24.013865715642822im,871.7666471902457,
                        775.9378880526116+72.28506946810577im,864.8042904109644+48.484435244566164im,
                        951.9251138422927+23.087316244456495im,1040.8551248015983,839.2435295850004+96.50386215826941im,
                        935.577291435982+72.02059506731321im,1032.1122984377455+47.67031081833441im,
                        1128.4973659133643+23.486137625752562im,1223.7231460822115,904.612736582062+120.40780863009967im,
                        1007.3729131734569+96.62571433514094im,1111.9771096277711+72.20655596676869im,
                        1216.72079371581+48.68201692665717im,1319.2446600868416+23.824247553429345im,
                        1423.9794343181684] rtol=2eps(elty)
                elseif tri === :L
                    @test vec(pmc) ≈ elty[584.0390995693658,647.3346451503741-23.59650895453973im,
                    712.0457403536856-48.19774365201118im,775.9378880526116-72.28506946810577im,
                    839.2435295850004-96.50386215826941im,904.612736582062-120.40780863009967im,719.780122523535,
                    792.4319851444229-24.013865715642822im,864.8042904109644-48.484435244566164im,
                    935.577291435982-72.02059506731321im,1007.3729131734569-96.62571433514094im,871.7666471902457,
                    951.9251138422927-23.087316244456495im,1032.1122984377455-47.67031081833441im,
                    1111.9771096277711-72.20655596676869im,1040.8551248015983,1128.4973659133643-23.486137625752562im,
                    1216.72079371581-48.68201692665717im,1223.7231460822115,1319.2446600868416-23.824247553429345im,
                    1423.9794343181684] rtol=2eps(elty)
                end
            end
        end

        @testset "Scalar product and norm" begin
            pms = SPMatrix(n, sparsevec(data), fmt)
            @test dot(pm, pm) ≈ 15.377768001340606 rtol=2eps(elty)
            @test dot(pm, pms) ≈ 15.377768001340606 rtol=2eps(elty)
            @test dot(pms, pms) ≈ 15.377768001340606 rtol=2eps(elty)
            for p in (pm, pms)
                @test norm(pm) ≈ sqrt(15.377768001340606) rtol=2eps(elty)
                @test norm(pm, 1) ≈ 20.750571954272097 rtol=2eps(elty)
                @test eqapprox(norm(pm, Inf), relty(0.9923158111197444), sca, rtol=2eps(elty))
                @test norm(pm, 0) == n^2
                @test eqapprox(norm(pm, -Inf), relty(0.020565681831590243), sca, rtol=2eps(elty))
                @test norm(pm, 3.) ≈ 2.3265784736459425 rtol=2eps(elty)
            end
            fill!(pmc, zero(elty))
            pmc[diagind(pmc)] .= elty.(3:8) .+ elty.(9:14) .* im
            pmcs = SPMatrix(n, SparseVector{elty,Int}(packedsize(n), Int[], elty[]), fmt)
            pmcs[diagind(pmcs)] .= elty.(3:8) + elty.(9:14) .* im
            @test dot(pm, pmc) ≈ 1.09927068388477 + 1.9607175946344464im rtol=2eps(elty)
            @test dot(pm, pmcs) ≈ 1.09927068388477 + 1.9607175946344464im rtol=2eps(elty)
            @test dot(pms, pmcs) ≈ 1.09927068388477 + 1.9607175946344464im rtol=2eps(elty)
            for p in (pmc, pmcs)
                @test norm(p) ≈ 31.78049716414141 rtol=2eps(elty)
                @test norm(p, 1) ≈ 76.64595499019796 rtol=2eps(elty)
                @test norm(p, Inf) == elty(16.1245154965971)
                @test norm(p, 0) == n
                @test norm(p, -Inf) == 0
                @test norm(p, 3.) ≈ 23.924156867384085 rtol=2eps(elty)
            end
        end

        pmc .= pm
        @test pmc == pm
        if elty <: LinearAlgebra.BlasFloat # we need LAPACK support for these tests
            @testset "Eigendecomposition" begin
                es = eigen!(pmc)
                @test es.values ≈ elty[-2.2593488980647485, -1.4425783744956544, -0.16598803850018665, 0.07505155001730082,
                    1.5148316538271211, 2.421606592341114]
                for (checkvec, refvec) in zip(eachcol(es.vectors), eachcol(elty[0.31445468639996366+0.027380089427519492im -0.356846859104041-0.3411553170041212im 0.30344743725666906-0.1796357044454669im -0.22409188361684867+0.32628390313342026im 0.40759509341406064+0.38479842962095157im 0.24231313048439662-0.05191352113022923im
                    0.5411044255265841-0.11459617342740933im -0.05538753083375486-0.05438144556497336im 0.046989238408721073+0.35598029145284327im 0.5338953789528786+0.2566668116412335im -0.17518754569748293+0.037401932349843785im -0.17742835414847183+0.38029739896099923im
                    -0.283175449639424-0.037754458263578775im 0.350706198289628-0.26234386054717546im 0.32200085027494607-0.43084123203404345im 0.2549147721973212+0.47449806416768875im -0.27676479494247985-0.1902683293622268im 0.17423710959632213+0.06300078489313002im
                    -0.3155950014491524+0.2098855118181834im -0.21742900918548894-0.31152581667534185im -0.13067784766898125+0.11766853912466742im 0.007879703803434938-0.06184213990943489im 0.35564499478811096-0.38230557716459246im 0.11151470809245143+0.6262096987086955im
                    0.4912221607748349+0.1774551826345571im 0.13957826117678163-0.1494878354737711im 0.2784647279974575-0.4239405504083408im -0.06875232544748669-0.43590201734315853im 0.05977247213382239-0.38442175560636926im -0.28364004752140015+0.039643231555456196im
                    0.31046236185258075 0.6101448168343749 -0.411365063286737 0.06037855103910823 0.3419243886295883 0.4914848442270254]))
                    # the phase can be arbitrary, so normalize first (we change the reference, so all further comparisons
                    # should work without normalization - although it is not really guaranteed that the different eigenvalue
                    # algorithms produce the same phases)
                    rmul!(refvec, cis(angle(checkvec[1]) - angle(refvec[1])))
                    @test ≈(checkvec, refvec, atol=50eps(elty))
                end
                @test eigvals(pm) ≈ es.values
                @test vec(pm) == data
                es2 = eigen(pm, relty(-.7), relty(1.))
                @test es2.values ≈ @view(es.values[3:4])
                for (checkvec, refvec) in zip(eachcol(es2.vectors), eachcol(@view(es.vectors[:, 3:4])))
                    @test ≈(checkvec, refvec, atol=16eps(elty)) || ≈(checkvec, -refvec, atol=16eps(elty))
                end
                es3 = eigen(pm, 3:5)
                @test es3.values ≈ @view(es.values[3:5]) rtol=4eps(elty)
                for (checkvec, refvec) in zip(eachcol(es3.vectors), eachcol(@view(es.vectors[:, 3:5])))
                    @test ≈(checkvec, refvec, atol=8eps(elty)) || ≈(checkvec, -refvec, atol=8eps(elty))
                end
                @test eigvals(pm, relty(-.7), relty(1.)) ≈ @view(es.values[3:4])
                @test eigvals(pm, 3:5) ≈ @view(es.values[3:5])
                @test eigmin(pm) ≈ minimum(es.values)
                @test eigmax(pm) ≈ maximum(es.values)
                @test eigvecs(pm) == es.vectors

                # functions that are not called by the LinearAlgebra interface
                @test spev!('N', copy(pm)) ≈ es.values
                let newev=spev!('V', copy(pm))
                    @test newev[1] ≈ es.values
                    @test newev[2] ≈ es.vectors
                end
                @test spevx!('N', copy(pm)) ≈ es.values
                @test spevx!('N', StandardPacked.packed_ulchar(pm), vec(packed_unscale!(copy(pm)))) ≈ es.values
                let newev=spevx!('V', copy(pm))
                    @test newev[1] ≈ es.values
                    @test newev[2] ≈ es.vectors
                end
                let newev=spevx!('V', StandardPacked.packed_ulchar(pm), vec(packed_unscale!(copy(pm))))
                    @test newev[1] ≈ es.values
                    @test newev[2] ≈ es.vectors
                end

                # check pre-allocated routines (eigenvalues only)
                # spevd
                @test_throws DimensionMismatch eigvals!(pm, Vector{relty}(undef, n +1))
                @test_throws DimensionMismatch eigvals!(pm, Vector{relty}(undef, n -1))
                @test_throws ArgumentError eigvals!(pm, missing, Vector{elty}(undef, n -1))
                @test_throws ArgumentError eigvals!(pm, missing, missing, Vector{relty}(undef, n -1))
                @test_throws ArgumentError eigvals!(pm, missing, missing, missing, Vector{Int}(undef, 0))
                @test eigvals(pm, Vector{relty}(undef, n), Vector{elty}(undef, n), Vector{relty}(undef, n),
                    Vector{Int}(undef, 1)) ≈ es.values
                # spev
                @test_throws DimensionMismatch spev!('N', pm, Vector{relty}(undef, n +1))
                @test_throws DimensionMismatch spev!('N', pm, Vector{relty}(undef, n -1))
                @test_throws ArgumentError spev!('N', pm, missing, Vector{elty}(undef, 2n -2))
                @test_throws ArgumentError spev!('N', pm, missing, missing, Vector{relty}(undef, 3n -3))
                @test spev!('N', copy(pm), Vector{relty}(undef, n), Vector{elty}(undef, 2n -1),
                    Vector{relty}(undef, 3n -2)) ≈ es.values
                # spevx interval range
                @test_throws ArgumentError eigvals!(pm, relty(-.7), relty(1.), missing, Vector{elty}(undef, 2n -1))
                @test_throws ArgumentError eigvals!(pm, relty(-.7), relty(1.), missing, missing, Vector{relty}(undef, 7n -1))
                @test_throws ArgumentError eigvals!(pm, relty(-.7), relty(1.), missing, missing, missing,
                    Vector{Int}(undef, 5n -1))
                evstore = fill(relty(42), n)
                @test eigvals(pm, relty(-.7), relty(1.), evstore, Vector{elty}(undef, 2n), Vector{relty}(undef, 7n),
                    Vector{Int}(undef, 5n)) ≈ @view(es.values[3:4])
                @test all(x -> x == relty(42.), @view(evstore[3:end]))
                # spevx index range
                @test_throws DimensionMismatch eigvals!(pm, 3:5, Vector{relty}(undef, 2))
                # spevx total
                @test_throws DimensionMismatch spevx!('N', 'A', pm, nothing, nothing, nothing, nothing, relty(-1),
                    Vector{relty}(undef, n -1))

                # check pre-allocated routines (including eigenvectors)
                # spevd
                @test_throws DimensionMismatch eigen!(pm, Vector{relty}(undef, n +1))
                @test_throws DimensionMismatch eigen!(pm, Vector{relty}(undef, n -1))
                @test_throws DimensionMismatch eigen!(pm, missing, Matrix{elty}(undef, 1, n))
                @test_throws DimensionMismatch eigen!(pm, missing, Matrix{elty}(undef, n -1, n -1))
                @test_throws DimensionMismatch eigen!(pm, missing, Matrix{elty}(undef, n +1, n +1))
                @test_throws ArgumentError eigen!(pm, missing, missing, Vector{elty}(undef, 2n -1))
                @test_throws ArgumentError eigen!(pm, missing, missing, missing, Vector{relty}(undef, 2n^2 + 5n))
                @test_throws ArgumentError eigen!(pm, missing, missing, missing, missing, Vector{Int}(undef, 2 + 5n))
                @test eigen(pm, Vector{relty}(undef, n), Matrix{elty}(undef, n, n), Vector{elty}(undef, 2n),
                    Vector{relty}(undef, 2n^2 + 5n +1), Vector{Int}(undef, 3 + 5n)) == es
                # spev
                @test_throws DimensionMismatch spev!('V', pm, Vector{relty}(undef, n +1))
                @test_throws DimensionMismatch spev!('V', pm, Vector{relty}(undef, n -1))
                @test_throws DimensionMismatch spev!('V', pm, missing, Matrix{elty}(undef, 1, n))
                @test_throws DimensionMismatch spev!('V', pm, missing, Matrix{elty}(undef, n -1, n -1))
                @test_throws DimensionMismatch spev!('V', pm, missing, Matrix{elty}(undef, n +1, n +1))
                @test_throws ArgumentError spev!('V', pm, missing, missing, Vector{elty}(undef, 2n -2))
                @test_throws ArgumentError spev!('V', pm, missing, missing, missing, Vector{relty}(undef, 3n -3))
                let newev=spev!('V', copy(pm), Vector{relty}(undef, n), Matrix{elty}(undef, n, n), Vector{elty}(undef, 2n -1),
                    Vector{relty}(undef, 3n -2))
                    @test newev[1] ≈ es.values
                    @test newev[2] ≈ es.vectors
                end
                # spevx interval range
                @test_throws DimensionMismatch eigen!(pm, relty(-.7), relty(1.), missing, Matrix{elty}(undef, 1, n))
                @test_throws DimensionMismatch eigen!(pm, relty(-.7), relty(1.), missing, Matrix{elty}(undef, n +1, n))
                @test_throws ArgumentError eigen!(pm, relty(-.7), relty(1.), missing, missing, Vector{elty}(undef, 2n -1))
                @test_throws ArgumentError eigen!(pm, relty(-.7), relty(1.), missing, missing, missing,
                    Vector{relty}(undef, 7n -1))
                @test_throws ArgumentError eigen!(pm, relty(-.7), relty(1.), missing, missing, missing, missing,
                    Vector{Int}(undef, 5n -1))
                @test_throws DimensionMismatch eigen!(pm, relty(-.7), relty(1.), missing, missing, missing, missing, missing,
                    Vector{Int}(undef, n -1))
                evstore = fill(relty(42), n)
                evecstore = fill(elty(42), n, n)
                @test eigen(pm, relty(-.7), relty(1.), evstore, evecstore, Vector{elty}(undef, 2n), Vector{relty}(undef, 7n),
                    Vector{Int}(undef, 5n), Vector{Int}(undef, n)) == es2
                @test all(x -> x == relty(42.), @view(evstore[3:end]))
                @test all(x -> x == elty(42.), @view(evecstore[:, 3:end]))
                # spevx index range
                @test_throws DimensionMismatch eigen!(pm, 3:5, Vector{relty}(undef, 2))
                @test_throws DimensionMismatch eigen!(pm, 3:5, missing, Matrix{elty}(undef, n, 2))
                # spevx total
                @test_throws DimensionMismatch spevx!('V', 'A', pm, nothing, nothing, nothing, nothing, relty(-1),
                    Vector{relty}(undef, n -1))
                @test_throws DimensionMismatch spevx!('V', 'A', pm, nothing, nothing, nothing, nothing, relty(-1),
                    missing, Matrix{elty}(undef, n, n -1))
            end

            @testset "Generalized eigendecomposition" begin
                if fmt === :U
                    bdata = elty[3.8033400135595294,1.6307440461120737+1.901241452323681im,5.095386274988267,
                        0.6988133745727712+0.830255332527524im,-0.15469365020862613+0.6392363948771735im,4.194011539474447,
                        -0.11057132391743774+0.9375964768047319im,0.7387697109368104+0.7222224591057691im,
                        -0.20282406157995014-1.893548126969197im,3.6817898120323993,-0.6802583973741567-0.3400716386737509im,
                        0.33424719032767364-1.5946424644538961im,-1.847619437677855+0.731740535284063im,
                        -0.11531915167576953-2.16985892752569im,3.67987308967989,-0.17153896566260285+1.8387604542416423im,
                        1.1959389376852303+0.47691337561622726im,0.7249232709872941+0.897637788119518im,
                        -0.22550571352535637-1.398097042063932im,0.6628018582324092-0.4937799494126702im,2.803213794390974]
                elseif fmt === :L
                    bdata = elty[3.8033400135595294,1.6307440461120737-1.901241452323681im,
                        0.6988133745727712-0.830255332527524im,-0.11057132391743774-0.9375964768047319im,
                        -0.6802583973741567+0.3400716386737509im,-0.17153896566260285-1.8387604542416423im,5.095386274988267,
                        -0.15469365020862613-0.6392363948771735im,0.7387697109368104-0.7222224591057691im,
                        0.33424719032767364+1.5946424644538961im,1.1959389376852303-0.47691337561622726im,4.194011539474447,
                        -0.20282406157995014+1.893548126969197im,-1.847619437677855-0.731740535284063im,
                        0.7249232709872941-0.897637788119518im,3.6817898120323993,-0.11531915167576953+2.16985892752569im,
                        -0.22550571352535637+1.398097042063932im,3.67987308967989,0.6628018582324092+0.4937799494126702im,
                        2.803213794390974]
                elseif fmt === :US
                    bdata = elty[3.8033400135595294,2.306220346770871+2.68876144722207im,5.095386274988267,
                        0.9882713518885228+1.1741583514930085im,-0.21876985813803865+0.9040167791977821im,4.194011539474447,
                        -0.15637146589358905+1.325961653530483im,1.0447781446772884+1.0213767967178267im,
                        -0.2868365386619613-2.6778814421660098im,3.6817898120323993,-0.9620306514847188-0.4809339235908613im,
                        0.4726969097464973-2.2551650003667563im,-2.612928466868174+1.0348373891368703im,
                        -0.16308590830123332-3.0686439237431697im,3.67987308967989,-0.24259273171550563+2.6003999723438436im,
                        1.6913130654445245+0.6744573638736028im,1.0251963215100979+1.2694515340572092im,
                        -0.3189132384601809-1.9772077984005203im,0.9373433770783627-0.6983103012872991im,2.803213794390974]
                elseif fmt === :LS
                    bdata = elty[3.8033400135595294,2.306220346770871-2.68876144722207im,
                        0.9882713518885228-1.1741583514930085im,-0.15637146589358905-1.325961653530483im,
                        -0.9620306514847188+0.4809339235908613im,-0.24259273171550563-2.6003999723438436im,5.095386274988267,
                        -0.21876985813803865-0.9040167791977821im,1.0447781446772884-1.0213767967178267im,
                        0.4726969097464973+2.2551650003667563im,1.6913130654445245-0.6744573638736028im,4.194011539474447,
                        -0.2868365386619613+2.6778814421660098im,-2.612928466868174-1.0348373891368703im,
                        1.0251963215100979-1.2694515340572092im,3.6817898120323993,-0.16308590830123332+3.0686439237431697im,
                        -0.3189132384601809+1.9772077984005203im,3.67987308967989,0.9373433770783627+0.6983103012872991im,
                        2.803213794390974]
                end
                pmb = SPMatrix(n, bdata, fmt)
                es = eigen(pm, pmb)
                @test es.values ≈ elty[-1.484039608505412, -0.7335312961620745, -0.08396534056233655, 0.01312285400345312,
                    0.42567784779823886, 6.9689832966188865]
                for (checkvec, refvec) in zip(eachcol(es.vectors), eachcol(elty[-0.438622702511817+0.5812970443080179im 0.3047342049768903+0.2149056923169168im 0.11935654276323548-0.10719373964797772im -0.01722140174978548-0.1529474734596131im 0.19948149102938326+0.013388337854375819im 1.2833810201351943-0.7284448357978738im
                    -0.2869246107316373-0.15705984996656544im -0.1393753982919387+0.11033224511630198im 0.048472986098988036+0.3306420560541399im -0.2499201853233792+0.07643662576760785im 0.005044937452898591+0.05328535061634155im -0.3691882982710521+0.6851115898642772im
                    0.3038498714925774-0.13501753285206108im 0.0129093405326802+0.3829824925451977im 0.16143026795866086-0.2703877896538269im -0.19651062988378934-0.06571227897275021im -0.21504949545930296-0.058065733618046754im -0.8726968029532245+0.05974022302214493im
                    -0.3998081942920746-0.7207410985073712im 0.22678215020968834+0.0032892144267649553im -0.06524183368131622+0.0657256261760097im 0.005571837399035091+0.01676633106743556im 0.029402710627367892-0.35454265887365605im 0.5139330279764214+1.4931482208078115im
                    -0.5211138374952795+0.5781342307063649im 0.06180067992198105+0.11001772016672837im 0.20713538687232932-0.3794133121752251im 0.16806712406247323+0.12475256043168143im -0.1635059948360116-0.30720184651212135im 0.5544256249481049-0.34577556210027477im
                    -0.349047911411371 -0.48411218115737703 -0.3000535252177965 -0.025523958596418112 -0.032698856317929355 1.50957957756881]))
                    rmul!(refvec, cis(angle(checkvec[1]) - angle(refvec[1])))
                    @test ≈(checkvec, refvec, atol=100eps(elty))
                end
                @test eigvals(pm, pmb) ≈ es.values
                @test vec(pm) == data && vec(pmb) == bdata
                es2 = spgvx!(1, 'V', 'V', copy(pm), copy(pmb), relty(-.7), relty(1.), nothing, nothing, relty(-1))
                @test es2[1] ≈ @view(es.values[3:5])
                for (checkvec, refvec) in zip(eachcol(es2[2]), eachcol(@view(es.vectors[:, 3:5])))
                    @test ≈(checkvec, refvec, atol=16eps(elty)) || ≈(checkvec, -refvec, atol=16eps(elty))
                end
                es3 = spgvx!(1, 'V', 'I', copy(pm), copy(pmb), nothing, nothing, 3, 5, relty(-1))
                @test es3[1] ≈ @view(es.values[3:5]) rtol=16eps(elty)
                for (checkvec, refvec) in zip(eachcol(es3[2]), eachcol(@view(es.vectors[:, 3:5])))
                    @test ≈(checkvec, refvec, atol=16eps(elty)) || ≈(checkvec, -refvec, atol=16eps(elty))
                end

                # functions that are not called by the LinearAlgebra interface
                @test spgv!(1, 'N', copy(pm), copy(pmb))[1] ≈ es.values
                let newev=spgv!(1, 'V', copy(pm), copy(pmb))
                    @test newev[1] ≈ es.values
                    @test newev[2] ≈ es.vectors
                end
                @test spgvx!(1, 'N', copy(pm), copy(pmb))[1] ≈ es.values
                @test spgvx!(1, 'N', StandardPacked.packed_ulchar(pm), vec(packed_unscale!(copy(pm))),
                    vec(packed_unscale!(copy(pmb))))[1] ≈ es.values
                let newev=spgvx!(1, 'V', copy(pm), copy(pmb))
                    @test newev[1] ≈ es.values
                    @test newev[2] ≈ es.vectors
                end
                let newev=spgvx!(1, 'V', StandardPacked.packed_ulchar(pm), vec(packed_unscale!(copy(pm))),
                        vec(packed_unscale!(copy(pmb))))
                    @test newev[1] ≈ es.values
                    @test newev[2] ≈ es.vectors
                end

                # check pre-allocated routines (eigenvalues only)
                # spgvd
                @test_throws DimensionMismatch spgvd!(1, 'N', pm, pmb, Vector{relty}(undef, n +1))
                @test_throws DimensionMismatch spgvd!(1, 'N', pm, pmb, Vector{relty}(undef, n -1))
                @test_throws ArgumentError spgvd!(1, 'N', pm, pmb, missing, Vector{elty}(undef, n -1))
                @test_throws ArgumentError spgvd!(1, 'N', pm, pmb, missing, missing, Vector{relty}(undef, n -1))
                @test_throws ArgumentError spgvd!(1, 'N', pm, pmb, missing, missing, missing, Vector{Int}(undef, 0))
                @test spgvd!(1, 'N', copy(pm), copy(pmb), Vector{relty}(undef, n), Vector{elty}(undef, n),
                    Vector{relty}(undef, n), Vector{Int}(undef, 1))[1] ≈ es.values
                # spgv
                @test_throws DimensionMismatch spgv!(1, 'N', pm, pmb, Vector{relty}(undef, n +1))
                @test_throws DimensionMismatch spgv!(1, 'N', pm, pmb, Vector{relty}(undef, n -1))
                @test_throws ArgumentError spgv!(1, 'N', pm, pmb, missing, Vector{elty}(undef, 2n -2))
                @test_throws ArgumentError spgv!(1, 'N', pm, pmb, missing, missing, Vector{relty}(undef, 3n -3))
                @test spgv!(1, 'N', copy(pm), copy(pmb), Vector{relty}(undef, n), Vector{elty}(undef, 2n -1),
                    Vector{relty}(undef, 3n -2))[1] ≈ es.values
                # spgvx interval range
                @test_throws ArgumentError spgvx!(1, 'N', 'V', pm, pmb, relty(-.7), relty(1.), nothing, nothing, relty(-1),
                    missing, Vector{elty}(undef, 2n -1))
                @test_throws ArgumentError spgvx!(1, 'N', 'V', pm, pmb, relty(-.7), relty(1.), nothing, nothing, relty(-1),
                    missing, missing, Vector{relty}(undef, 7n -1))
                @test_throws ArgumentError spgvx!(1, 'N', 'V', pm, pmb, relty(-.7), relty(1.), nothing, nothing, relty(-1),
                    missing, missing, missing, Vector{Int}(undef, 5n -1))
                evstore = fill(relty(42), n)
                @test spgvx!(1, 'N', 'V', copy(pm), copy(pmb), relty(-.7), relty(1.), nothing, nothing, relty(-1), evstore,
                    Vector{elty}(undef, 2n), Vector{relty}(undef, 7n), Vector{Int}(undef, 5n))[1] ≈ @view(es.values[3:5])
                @test all(x -> x == relty(42.), @view(evstore[4:end]))
                # spgvx index range
                @test_throws DimensionMismatch spgvx!(1, 'N', 'I', pm, pmb, nothing, nothing, 3, 5, relty(-1),
                    Vector{relty}(undef, 2))
                # spevx total
                @test_throws DimensionMismatch spgvx!(1, 'N', 'A', pm, pmb, nothing, nothing, nothing, nothing, relty(-1),
                    Vector{relty}(undef, n -1))

                # check pre-allocated routines (including eigenvectors)
                # spgvd
                @test_throws DimensionMismatch spgvd!(1, 'V', pm, pmb, Vector{relty}(undef, n +1))
                @test_throws DimensionMismatch spgvd!(1, 'V', pm, pmb, Vector{relty}(undef, n -1))
                @test_throws DimensionMismatch spgvd!(1, 'V', pm, pmb, missing, Matrix{elty}(undef, 1, n))
                @test_throws DimensionMismatch spgvd!(1, 'V', pm, pmb, missing, Matrix{elty}(undef, n -1, n -1))
                @test_throws DimensionMismatch spgvd!(1, 'V', pm, pmb, missing, Matrix{elty}(undef, n +1, n +1))
                @test_throws ArgumentError spgvd!(1, 'V', pm, pmb, missing, missing, Vector{elty}(undef, 2n -1))
                @test_throws ArgumentError spgvd!(1, 'V', pm, pmb, missing, missing, missing, Vector{relty}(undef, 2n^2 + 5n))
                @test_throws ArgumentError spgvd!(1, 'V', pm, pmb, missing, missing, missing, missing,
                    Vector{Int}(undef, 2 + 5n))
                let newev=spgvd!(1, 'V', copy(pm), copy(pmb), Vector{relty}(undef, n), Matrix{elty}(undef, n, n),
                    Vector{elty}(undef, 2n), Vector{relty}(undef, 2n^2 + 5n +1), Vector{Int}(undef, 3 + 5n))
                    @test newev[1] == es.values
                    @test newev[2] == es.vectors
                end
                # spgv
                @test_throws DimensionMismatch spgv!(1, 'V', pm, pmb, Vector{relty}(undef, n +1))
                @test_throws DimensionMismatch spgv!(1, 'V', pm, pmb, Vector{relty}(undef, n -1))
                @test_throws DimensionMismatch spgv!(1, 'V', pm, pmb, missing, Matrix{elty}(undef, 1, n))
                @test_throws DimensionMismatch spgv!(1, 'V', pm, pmb, missing, Matrix{elty}(undef, n -1, n -1))
                @test_throws DimensionMismatch spgv!(1, 'V', pm, pmb, missing, Matrix{elty}(undef, n +1, n +1))
                @test_throws ArgumentError spgv!(1, 'V', pm, pmb, missing, missing, Vector{elty}(undef, 2n -2))
                @test_throws ArgumentError spgv!(1, 'V', pm, pmb, missing, missing, missing, Vector{relty}(undef, 3n -3))
                let newev=spgv!(1, 'V', copy(pm), copy(pmb), Vector{relty}(undef, n), Matrix{elty}(undef, n, n),
                    Vector{elty}(undef, 2n -1), Vector{relty}(undef, 3n -2))
                    @test newev[1] ≈ es.values
                    @test newev[2] ≈ es.vectors
                end
                # spgvx interval range
                @test_throws DimensionMismatch spgvx!(1, 'V', 'V', pm, pmb, relty(-.7), relty(1.), nothing, nothing, relty(-1),
                    missing, Matrix{elty}(undef, 1, n))
                @test_throws DimensionMismatch spgvx!(1, 'V', 'V', pm, pmb, relty(-.7), relty(1.), nothing, nothing, relty(-1),
                    missing, Matrix{elty}(undef, n +1, n))
                @test_throws ArgumentError spgvx!(1, 'V', 'V', pm, pmb, relty(-.7), relty(1.), nothing, nothing, relty(-1),
                    missing, missing, Vector{elty}(undef, 2n -1))
                @test_throws ArgumentError spgvx!(1, 'V', 'V', pm, pmb, relty(-.7), relty(1.), nothing, nothing, relty(-1),
                    missing, missing, missing, Vector{relty}(undef, 7n -1))
                @test_throws ArgumentError spgvx!(1, 'V', 'V', pm, pmb, relty(-.7), relty(1.), nothing, nothing, relty(-1),
                    missing, missing, missing, missing, Vector{Int}(undef, 5n -1))
                @test_throws DimensionMismatch spgvx!(1, 'V', 'V', pm, pmb, relty(-.7), relty(1.), nothing, nothing, relty(-1),
                    missing, missing, missing, missing, missing, Vector{Int}(undef, n -1))
                evstore = fill(relty(42), n)
                evecstore = fill(elty(42), n, n)
                @test spgvx!(1, 'V', 'V', copy(pm), copy(pmb), relty(-.7), relty(1.), nothing, nothing, relty(-1), evstore,
                    evecstore, Vector{elty}(undef, 2n), Vector{relty}(undef, 7n), Vector{Int}(undef, 5n),
                    Vector{Int}(undef, n)) == es2
                @test all(x -> x == relty(42.), @view(evstore[4:end]))
                @test all(x -> x == elty(42.), @view(evecstore[:, 4:end]))
                # spegx index range
                @test_throws DimensionMismatch spgvx!(1, 'V', 'I', pm, pmb, nothing, nothing, 3, 5, relty(-1),
                    Vector{relty}(undef, 2))
                @test_throws DimensionMismatch spgvx!(1, 'V', 'I', pm, pmb, nothing, nothing, 3, 5, relty(-1), missing,
                    Matrix{elty}(undef, n, 2))
                # spgvx total
                @test_throws DimensionMismatch spgvx!(1, 'V', 'A', pm, pmb, nothing, nothing, nothing, nothing, relty(-1),
                    Vector{relty}(undef, n -1))
                @test_throws DimensionMismatch spgvx!(1, 'V', 'A', pm, pmb, nothing, nothing, nothing, nothing, relty(-1),
                    missing, Matrix{elty}(undef, n, n -1))
            end

            rhs = elty[1.0 8.0; 2.0 9.0; 3.0 10.0; 4.0 11.0; 5.0 12.0; 6.0 13.0]
            rhs .+= 2im .* rhs
            @testset "Cholesky decomposition" begin
                copyto!(pmc, pm)
                @test !LinearAlgebra.isposdef(pmc)
                @test_throws PosDefException cholesky(pmc)
                @test pmc == pm
                @test LinearAlgebra.isposdef(pmc, relty(3))
                chol = cholesky!(pmc, shift=relty(3))
                @test chol.U ≈ elty[1.743301342099473 -0.3816637052689344-0.23145226571921448im 0.026237778048448325+0.11343056259741721im -0.035628921912937914+0.16352277212295752im -0.43392980704566214+0.28902757435074533im 0.3514805887340323+0.23392893715583205im
                    0 1.6065133638749758 0.2914715328582404+0.03179883368945122im 0.5157379199874405+0.3455261305971578im -0.32457105355482685+0.14400156542154582im -0.27313564272898627+0.39442282628652703im
                    0 0 1.6331336309153945 -0.1554129186067864-0.6155814723017263im 0.1107829565155038-0.2686778686114766im 0.005156923839459556+0.0714200046468089im
                    0 0 0 1.743171836386023 0.23182118053900638-0.4996001056318342im 0.4270452091439025+0.26271501636548444im
                    0 0 0 0 1.3919156801166248 -0.5496104112521715-0.12035374245116845im
                    0 0 0 0 0 1.4134814084964848]
                if fmt === :U
                    @test inv(chol) ≈ SPMatrix(n, elty[0.46676198444101286, 0.19669555770270547+0.06898329440739434im,
                        0.6161220314750114, -0.08497237300850695-0.05491894824021812im,
                        -0.16138772611064955+0.040912777231089686im, 0.46110669589135866,
                        -0.03824811431135422-0.09674411265061586im, -0.2123024038723344-0.09853816136876509im,
                        0.07702279672313024+0.15869603128794338im, 0.43380773330117595,
                        0.1909017652266846-0.0925836534293675im, 0.24128074253952864-0.13690821120765917im,
                        -0.11323562173751334+0.05366412351760558im, -0.1344564610265297+0.15218514529491906im,
                        0.5979275473532369, -0.003600211931421843-0.06525873396755538im,
                        0.17942885520418508-0.07866489028744082im, -0.02820467474351798-0.05544700421341201im,
                        -0.16130439852103787-0.02454626672741417im, 0.19763407368114647+0.04327792908654072im,
                        0.5005181132856956]) rtol=150eps(elty)
                elseif fmt === :L
                    @test inv(chol) ≈ SPMatrix(n, elty[0.46676198444101286, 0.19669555770270547-0.06898329440739431im,
                        -0.08497237300850694+0.05491894824021812im, -0.03824811431135421+0.09674411265061583im,
                        0.19090176522668456+0.0925836534293675im, -0.0036002119314218345+0.06525873396755541im,
                        0.6161220314750114, -0.16138772611064953-0.040912777231089686im,
                        -0.21230240387233437+0.09853816136876509im, 0.2412807425395286+0.13690821120765917im,
                        0.17942885520418506+0.07866489028744088im, 0.46110669589135866,
                        0.07702279672313024-0.15869603128794338im, -0.11323562173751334-0.053664123517605584im,
                        -0.02820467474351798+0.05544700421341198im, 0.43380773330117595,
                        -0.1344564610265297-0.15218514529491906im, -0.16130439852103784+0.024546266727414154im,
                        0.5979275473532369, 0.19763407368114647-0.043277929086540684im, 0.5005181132856956], :L) rtol=50eps(elty)
                end
                @test rhs ≈ (pm + 3I) * (chol \ rhs)
            end

            @testset "Bunch-Kaufman decomposition" begin
                bk = bunchkaufman(pm)
                if fmt === :U
                    let P=bk.P, U=bk.U, D=bk.D
                        @test P' * U * D * U' * P ≈ Matrix(pm)
                    end
                    @test inv(bk) ≈ SPMatrix(n, elty[1.3581916719708789, -0.342956324395737+3.6781189369192617im,
                        3.8533309731653227, 0.20466871121679153+2.202713822098123im, 4.362865375776229-3.3131720125104414im,
                        2.0423406471237833, -0.015907782720826935+0.11013170779058595im,
                        -0.26505866835585146+0.8464254222169064im, 0.16221891050532888-0.1545875497136462im,
                        0.048913487296107755, -2.8397949519940435-1.8646232948832293im,
                        -1.2477730273120455+2.142342004160346im, -4.608600834355129+0.833299034525623im,
                        1.0208490147119043+0.22101859427166115im, 1.0288472918447749,
                        0.8206483322804747+0.034156080112838216im, 0.4194869057328578+1.21308087384012im,
                        0.8665574089133588-0.6000277739211239im, -0.079279901782069+0.3855856639808169im,
                        0.4641925210427782-1.4412075859084683im, -1.094699662661824])
                elseif fmt === :L
                    let P=bk.P, L=bk.L, D=bk.D
                        @test P' * L * D * L' * P ≈ Matrix(pm)
                    end
                    @test inv(bk) ≈ SPMatrix(n, elty[1.3581916719708789, -0.3429563243957337-3.6781189369192617im,
                        0.20466871121679364-2.2027138220981217im, -0.01590778272082699-0.11013170779058656im,
                        -2.8397949519940457+1.8646232948832278im, 0.820648332280475-0.034156080112838653im,
                        3.8533309731653227, 4.3628653757762255+3.3131720125104445im, -0.26505866835585-0.846425422216907im,
                        -1.2477730273120449-2.1423420041603474im, 0.4194869057328594-1.2130808738401189im, 2.0423406471237833,
                        0.16221891050532955+0.15458754971364544im, -4.60860083435513-0.8332990345256264im,
                        0.86655740891336+0.6000277739211242im, 0.048913487296107755, 1.0208490147119043-0.2210185942716605im,
                        -0.079279901782069-0.38558566398081684im, 1.0288472918447749, 0.4641925210427769+1.441207585908469im,
                        -1.094699662661824], :L)
                end
                @test rhs ≈ pm * (bk \ rhs)
            end
        end
    end
end end

@testset "Broadcasting" begin
    @testset "Same format" begin
        output = SPMatrix(2, [15., 27., 31.], :U) .+ SPMatrix(2, [11., 17., 32.], :U)
        @test output isa SPMatrix{Float64,Vector{Float64},:U}
        @test output == [15+11 27+17; 27+17 31+32]
    end
    @testset "Generic matrix" begin
        output = [1. 2.; 3. 4.] .+ SPMatrix(2, [15., 27., 31.], :U)
        @test output isa Matrix
        @test output == [1+15 2+27; 3+27 4+31]
    end
    @testset "Different formats" begin
        output = SPMatrix(2, [15., 27., 31.], :U) .+ packed_scale!(SPMatrix(2, [11., 17., 32.], :U))
        @test output isa Matrix
        @test output == [15+11 27+17; 27+17 31+32]

        output = SPMatrix(2, [15., 27., 31.], :U) .+ SPMatrix(2, [11., 17., 32.], :L)
        @test output isa Matrix
        @test output == [15+11 27+17; 27+17 31+32]
    end
    @testset "Copy into matrix" begin
        test = SPMatrix{Float64}(undef, 2, :U)
        test .= [14. 8.; 19. 20.]
        @test test == [14. 8.; 8. 20.]

        scoutput = SPMatrix{Float64}(undef, 2, :U)
        scoutput .= 3 .* test .+ SPMatrix(2, [11., 17., 32.], :U)
        @test scoutput ≈ [3*14.0+11. 3*8.0+17.; 3*8.0+17. 3*20.0+32.]

        test = SPMatrix{Float64}(undef, 2, :US)
        test .= [14. 8.; 19. 20.]
        @test test ≈ [14. 8.; 8. 20.]

        test = SPMatrix{Float64}(undef, 2, :L)
        test .= [14. 8.; 19. 20.]
        @test test == [14. 19.; 19. 20.]

        test = SPMatrix{Float64}(undef, 2, :LS)
        test .= [14. 8.; 19. 20.]
        @test test ≈ [14. 19.; 19. 20.]
    end
    @testset "Combination with vector" begin
        @test SPMatrix(2, [15., 27., 31.], :U) .+ [1, 2, 3] == [16., 29., 34.]

        @test_throws DimensionMismatch SPMatrix(2, [15., 27., 31.], :U) .+ SPMatrix(2, [11., 17., 32.], :U) .+ [1, 2, 3]
        @test_throws DimensionMismatch SPMatrix(2, [15., 27., 31.], :U) .+ [1, 2, 3] .+ SPMatrix(2, [11., 17., 32.], :U)

        output = SPMatrix{Float64}(undef, 2, :U)
        output .= SPMatrix(2, [15., 27., 31.], :U) .+ [1, 2, 3]
        @test output == [16. 29.; 29. 34.]

        output = SPMatrix{Float64}(undef, 2, :L)
        @test_throws DimensionMismatch (output .= SPMatrix(2, [15., 27., 31.], :U) .+ [1, 2, 3])
    end
    @testset "Equality" begin
        mdata = rand(3, 3)
        mdata .+= mdata'
        for fmt_a in (:U, :US, :L, :LS), fmt_b in (:U, :US, :L, :LS)
            sma = SPMatrix{Float64}(undef, 3, fmt_a)
            smb = SPMatrix{Float64}(undef, 3, fmt_b)
            copyto!(sma, mdata)
            copyto!(smb, mdata)
            @test sma == smb skip=(fmt_a == :U && fmt_b == :LS ||
                                    fmt_a == :US && fmt_b == :L ||
                                    fmt_a == :L && fmt_b == :US ||
                                    fmt_a == :LS && fmt_b == :U) # these cases fall back to the generic routine and we don't
                                                                # have floating point equality
        end
    end
end

@testset "StaticArrays broadcasting" begin
    base = collect(1:16)
    pm = SPMatrix(4, @view(base[1:packedsize(4)]))
    s = SMatrix{4,4}([100 200 300 400; 500 600 700 800; 900 1000 1100 1200; 1300 1400 1500 1600])
    pm .= s
    @test base == [100, 200, 600, 300, 700, 1100, 400, 800, 1200, 1600, 11, 12, 13, 14, 15, 16]
    copyto!(base, 1:16)
    pm2 = SPMatrix(4, @view(base[1:packedsize(4)]), :L)
    pm2 .= s
    @test base == [100, 500, 900, 1300, 600, 1000, 1400, 1100, 1500, 1600, 11, 12, 13, 14, 15, 16]
end

@testset "Reshaping" begin
    pm = SPMatrix(6, 1:21)
    resh = reshape(pm, (2, 3, 3, 2))
    for l in 1:2, k in 1:3, j in 1:3, i in 1:2
        @test resh[i, j, k, l] == pm[(i-1)+2*(j-1)+1,(k-1)+3*(l-1)+1]
    end
end