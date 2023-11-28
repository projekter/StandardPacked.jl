using Test
using PackedMatrices
using LinearAlgebra, SparseArrays

setprecision(8)

@testset "Element type $elty" for elty in (Float32, Float64, BigFloat) begin
    refmatrix = elty[0.9555114416827896 0.14061115800771273 0.660907232019832 0.7388163447790721 0.15025179696269242 0.9847133877489483 0.3610824315069938
        0.14061115800771273 0.5391622763612515 0.21722146893315186 0.8336423569126538 0.41116243025455734 0.5343466059451054 0.43066682904413156
        0.660907232019832 0.21722146893315186 0.0019306390378336369 0.22609164615603128 0.35352758268808504 0.644850818020348 0.8100978477428766
        0.7388163447790721 0.8336423569126538 0.22609164615603128 0.5535820925972024 0.40253776414772224 0.2036884435817956 0.26796888784023765
        0.15025179696269242 0.41116243025455734 0.35352758268808504 0.40253776414772224 0.7741030577273389 0.33109536531255035 0.27758928139887895
        0.9847133877489483 0.5343466059451054 0.644850818020348 0.2036884435817956 0.33109536531255035 0.14756643458657948 0.09214560319184117
        0.3610824315069938 0.43066682904413156 0.8100978477428766 0.26796888784023765 0.27758928139887895 0.09214560319184117 0.45340576959493983]
    n = 7
    @test packedsize(refmatrix) == 28
    @testset "Upper unscaled" begin
        data = elty[0.9555114416827896,0.14061115800771273,0.5391622763612515,0.660907232019832,0.21722146893315186,
            0.0019306390378336369,0.7388163447790721,0.8336423569126538,0.22609164615603128,0.5535820925972024,
            0.15025179696269242,0.41116243025455734,0.35352758268808504,0.40253776414772224,0.7741030577273389,
            0.9847133877489483,0.5343466059451054,0.644850818020348,0.2036884435817956,0.33109536531255035,
            0.14756643458657948,0.3610824315069938,0.43066682904413156,0.8100978477428766,0.26796888784023765,
            0.27758928139887895,0.09214560319184117,0.45340576959493983]
        pm = PackedMatrix(n, data)
        @test size(pm) == (n, n)
        @test LinearAlgebra.checksquare(pm) == packedside(pm) == packedside(vec(pm)) == n
        @test eltype(pm) === elty
        @test packed_isupper(pm)
        @test !packed_islower(pm)
        @test !packed_isscaled(pm)
        @test issymmetric(pm)
        @test ishermitian(pm)
        @test transpose(pm) == pm

        @test collect(pm) == data
        @test [x for x in pm] == data
        @test Matrix(pm) == refmatrix
        @test PackedMatrix(Symmetric(refmatrix)) == pm
        @test [pm[i, j] for i in 1:n, j in 1:n] == refmatrix
        for i in 1:n, j in 1:n
            pm[i, j] = refmatrix[i, j]
        end
        @test Matrix(pm) == refmatrix
        fill!(pm, zero(elty))
        @test copyto!(pm, refmatrix) === pm
        @test Matrix(pm) == refmatrix
        @test_throws ErrorException PackedMatrix(pm, :L)
        @test diag(pm) == diag(refmatrix)
        @test tr(pm) == tr(refmatrix)

        pmc = copy(pm)
        @test pmc == pm
        @test packed_unscale!(pmc) === pmc && pmc == pm
        scaled = packed_scale!(pmc)
        @test packed_isscaled(scaled)
        @test Matrix(scaled) ≈ refmatrix rtol=2eps(elty)
        @test packed_unscale!(scaled) ≈ pmc rtol=4eps(elty)
        fill!(pmc, 5.)
        @test all(x -> x == 5., pmc)

        @test vec(2 .* pm) == 2data
        @test vec(pm .+ pm) == data .+ data
        @test vec(pm .+ data) == data .+ data
        pmc .= pm .+ 3 .* pm
        @test vec(pmc) == 4data
        @test_throws ErrorException pm .+ scaled

        if elty ∈ (Float32, Float64) # we need BLAS/LAPACK support for these tests
            tmpout = Vector{elty}(undef, n)
            @test mul!(tmpout, pm, elty.(collect(1:n)), true, false) === tmpout
            @test tmpout ≈ elty[15.361837164730108,13.481729135432627,13.312936427386411,10.409306064572332,
                11.443522912431101,7.988607484452503,9.839245597494859] rtol=2eps(elty)
            copyto!(pmc, pm)
            @test spr!(elty(4), elty.(collect(5:11)), pmc) === pmc
            @test vec(pmc) ≈ elty[100.95551144168279,120.14061115800772,144.53916227636125,140.66090723201984,
                168.21722146893316,196.00193063903782,160.73881634477908,192.83364235691266,224.22609164615602,
                256.5535820925972,180.1502517969627,216.41116243025456,252.3535275826881,288.4025377641477,
                324.77410305772736,200.98471338774894,240.5343466059451,280.64485081802036,320.20368844358177,
                360.33109536531254,400.14756643458657,220.361082431507,264.4306668290441,308.8100978477429,
                352.26796888784025,396.2775892813989,440.09214560319185,484.45340576959495] rtol=2eps(elty)
        end

        pms = PackedMatrix(n, sparsevec(data))
        @test dot(pm, pm) ≈ 12.788602219279593 rtol=2eps(elty)
        @test dot(pm, pms) ≈ 12.788602219279593 rtol=2eps(elty)
        @test dot(pms, pms) ≈ 12.788602219279593 rtol=2eps(elty)
        for p in (pm, pms)
            @test norm(pm) ≈ sqrt(12.788602219279593) rtol=2eps(elty)
            @test norm(pm, 1) ≈ 21.571292275978372 rtol=2eps(elty)
            @test norm(pm, Inf) == elty(0.9847133877489483)
            @test norm(pm, 0) == 49
            @test norm(pm, -Inf) == elty(0.0019306390378336369)
            @test norm(pm, 3.) ≈ 2.0766959423281186 rtol=2eps(elty)
        end

        fill!(pmc, zero(elty))
        pmc[diagind(pmc)] .= elty.(3:9)
        pmcs = PackedMatrix(n, SparseVector{elty,Int}(undef, packedsize(n)))
        pmcs[diagind(pmcs)] .= elty.(3:9)
        @test dot(pm, pmc) ≈ 19.034233988404225 rtol=2eps(elty)
        @test dot(pm, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
        @test dot(pms, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
        for p in (pm, pms)
            @test norm(pmcs) ≈ 16.73320053068151 rtol=2eps(elty)
            @test norm(pmcs, 1) ≈ 42 rtol=2eps(elty)
            @test norm(pmcs, Inf) == 9
            @test norm(pmcs, 0) == n
            @test norm(pmcs, -Inf) == 0
            @test norm(pmcs, 3.) ≈ 12.632719195312756 rtol=2eps(elty)
        end

        pmc .= pm
        @test pmc == pm
        if elty ∈ (Float32, Float64) # we need LAPACK support for these tests
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
                @test ≈(checkvec, refvec, atol=8eps(elty)) || ≈(checkvec, -refvec, atol=8eps(elty))
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
            @test_throws DimensionMismatch eigvals!(pm, 3:5, Vector{elty}(undef, 2))
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
            @test_throws DimensionMismatch eigen!(pm, 3:5, Vector{elty}(undef, 2))
            @test_throws DimensionMismatch eigen!(pm, 3:5, missing, Matrix{elty}(undef, n, 2))
            # spevx total
            @test_throws DimensionMismatch spevx!('V', 'A', pm, nothing, nothing, nothing, nothing, elty(-1),
                Vector{elty}(undef, n -1))
            @test_throws DimensionMismatch spevx!('V', 'A', pm, nothing, nothing, nothing, nothing, elty(-1),
                missing, Matrix{elty}(undef, n, n -1))

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
            @test inv(chol) ≈ PackedMatrix(n, elty[15.493628236589293, 18.055791818905497, 22.83884746037381,
                22.56819523083977, 28.380477703283688, 39.43723994306926, -13.664618497320284, -16.691276904083473,
                -20.71296414574622, 13.085254087036088, 1.6509049892926726, 1.8105792940720054, 2.1679624382488236,
                -1.5390396304367675, 0.8211060409222076, -31.02109455183003, -37.87395046635378, -49.32438789295014,
                27.97393922067559, -3.29559139381791, 65.7991061056096, -17.607661895505345, -21.93913378183722,
                -29.465880029842584, 15.993513682029798, -1.819156241457544, 37.72200796466474, 22.99414125333774]) rtol=150eps(elty)
            rhs = elty[1.0 8.0; 2.0 9.0; 3.0 10.0; 4.0 11.0; 5.0 12.0; 6.0 13.0; 7.0 14.0]
            @test rhs ≈ (pm + I) * (chol \ rhs)

            bk = bunchkaufman(pm)
            let P=bk.P, U=bk.U, D=bk.D
                @test P' * U * D * U' * P ≈ Matrix(pm)
            end
            @test inv(bk) ≈ PackedMatrix(n, elty[11.676181149869125, -25.74304267079464, 55.837437314564454,
                -3.890135969716243, 9.077199222177283, 1.0285759205945262, 24.102856515354063, -51.37534555845253,
                -9.226894089251159,  49.46819607136643, 9.176107458680693, -21.095279400248227, -3.294216491388214,
                19.103438949417768, 9.400381615980521, -26.29273501144287, 59.09033629509751, 9.912693866880879,
                -55.950010085611865, -21.636130300323575, 62.79650629970832, 7.584280750861903, -17.484312579760637,
                -1.9062117598362571, 16.528078653765096, 5.966940819148872, -19.347475787322477, 6.689363759583235])
            @test rhs ≈ pm * (bk \ rhs)
        end
    end
    @testset "Lower unscaled" begin
        data = elty[0.9555114416827896,0.14061115800771273,0.660907232019832,0.7388163447790721,0.15025179696269242,
            0.9847133877489483,0.3610824315069938,0.5391622763612515,0.21722146893315186,0.8336423569126538,
            0.41116243025455734,0.5343466059451054,0.43066682904413156,0.0019306390378336369,0.22609164615603128,
            0.35352758268808504,0.644850818020348,0.8100978477428766,0.5535820925972024,0.40253776414772224,0.2036884435817956,
            0.26796888784023765,0.7741030577273389,0.33109536531255035,0.27758928139887895,0.14756643458657948,
            0.09214560319184117,0.45340576959493983]
        pm = PackedMatrix(n, data, :L)
        @test size(pm) == (n, n)
        @test LinearAlgebra.checksquare(pm) == packedside(pm) == packedside(vec(pm)) == n
        @test eltype(pm) === elty
        @test !packed_isupper(pm)
        @test packed_islower(pm)
        @test !packed_isscaled(pm)
        @test issymmetric(pm)
        @test ishermitian(pm)
        @test transpose(pm) == pm

        @test collect(pm) == data
        @test [x for x in pm] == data
        @test Matrix(pm) == refmatrix
        @test PackedMatrix(Symmetric(refmatrix, :L)) == pm
        @test [pm[i, j] for i in 1:n, j in 1:n] == refmatrix
        for i in 1:n, j in 1:n
            pm[i, j] = refmatrix[i, j]
        end
        @test Matrix(pm) == refmatrix
        fill!(pm, zero(elty))
        @test copyto!(pm, refmatrix) === pm
        @test Matrix(pm) == refmatrix
        @test_throws ErrorException PackedMatrix(pm, :U)
        @test diag(pm) == diag(refmatrix)
        @test tr(pm) == tr(refmatrix)

        pmc = copy(pm)
        @test pmc == pm
        @test packed_unscale!(pmc) === pmc && pmc == pm
        scaled = packed_scale!(pmc)
        @test packed_isscaled(scaled)
        @test Matrix(scaled) ≈ refmatrix rtol=2eps(elty)
        @test packed_unscale!(scaled) ≈ pmc rtol=4eps(elty)
        fill!(pmc, 5.)
        @test all(x -> x == 5., pmc)

        @test vec(2 .* pm) == 2data
        @test vec(pm .+ pm) == data .+ data
        @test vec(pm .+ data) == data .+ data
        pmc .= pm .+ 3 .* pm
        @test vec(pmc) == 4data
        @test_throws ErrorException pm .+ scaled

        if elty ∈ (Float32, Float64) # we need BLAS/LAPACK support for these tests
            tmpout = Vector{elty}(undef, n)
            @test mul!(tmpout, pm, elty.(collect(1:n)), true, false) === tmpout
            @test tmpout ≈ elty[15.361837164730108,13.481729135432627,13.312936427386411,10.409306064572332,
                11.443522912431101,7.988607484452503,9.839245597494859] rtol=2eps(elty)
            copyto!(pmc, pm)
            @test spr!(elty(4), elty.(collect(5:11)), pmc) === pmc
            @test vec(pmc) ≈ elty[100.95551144168279,120.14061115800772,140.66090723201984,160.73881634477908,
                180.1502517969627,200.98471338774894,220.361082431507,144.53916227636125,168.21722146893316,
                192.83364235691266,216.41116243025456,240.5343466059451,264.4306668290441,196.00193063903782,
                224.22609164615602,252.3535275826881,280.64485081802036,308.8100978477429,256.5535820925972,
                288.4025377641477,320.20368844358177,352.26796888784025,324.77410305772736,360.33109536531254,
                396.2775892813989,400.14756643458657,440.09214560319185,484.45340576959495] rtol=2eps(elty)
        end

        pms = PackedMatrix(n, sparsevec(data), :L)
        @test dot(pm, pm) ≈ 12.788602219279593 rtol=2eps(elty)
        @test dot(pm, pms) ≈ 12.788602219279593 rtol=2eps(elty)
        @test dot(pms, pms) ≈ 12.788602219279593 rtol=2eps(elty)
        for p in (pm, pms)
            @test norm(pm) ≈ sqrt(12.788602219279593) rtol=2eps(elty)
            @test norm(pm, 1) ≈ 21.571292275978372 rtol=2eps(elty)
            @test norm(pm, Inf) == elty(0.9847133877489483)
            @test norm(pm, 0) == 49
            @test norm(pm, -Inf) == elty(0.0019306390378336369)
            @test norm(pm, 3.) ≈ 2.0766959423281186 rtol=2eps(elty)
        end

        fill!(pmc, zero(elty))
        pmc[diagind(pmc)] .= elty.(3:9)
        pmcs = PackedMatrix(n, SparseVector{elty,Int}(undef, packedsize(n)), :L)
        pmcs[diagind(pmcs)] .= elty.(3:9)
        @test dot(pm, pmc) ≈ 19.034233988404225 rtol=2eps(elty)
        @test dot(pm, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
        @test dot(pms, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
        for p in (pm, pms)
            @test norm(pmcs) ≈ 16.73320053068151 rtol=2eps(elty)
            @test norm(pmcs, 1) ≈ 42 rtol=2eps(elty)
            @test norm(pmcs, Inf) == 9
            @test norm(pmcs, 0) == n
            @test norm(pmcs, -Inf) == 0
            @test norm(pmcs, 3.) ≈ 12.632719195312756 rtol=2eps(elty)
        end

        pmc .= pm
        @test pmc == pm
        if elty ∈ (Float32, Float64) # we need LAPACK support for these tests
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
                @test ≈(checkvec, refvec, atol=8eps(elty)) || ≈(checkvec, -refvec, atol=8eps(elty))
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
            @test_throws DimensionMismatch eigvals!(pm, 3:5, Vector{elty}(undef, 2))
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
            @test_throws DimensionMismatch eigen!(pm, 3:5, Vector{elty}(undef, 2))
            @test_throws DimensionMismatch eigen!(pm, 3:5, missing, Matrix{elty}(undef, n, 2))
            # spevx total
            @test_throws DimensionMismatch spevx!('V', 'A', pm, nothing, nothing, nothing, nothing, elty(-1),
                Vector{elty}(undef, n -1))
            @test_throws DimensionMismatch spevx!('V', 'A', pm, nothing, nothing, nothing, nothing, elty(-1),
                missing, Matrix{elty}(undef, n, n -1))

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
            @test inv(chol) ≈ PackedMatrix(n, elty[15.493628236589293, 18.05579181890549, 22.568195230839763,
                -13.664618497320282, 1.6509049892926717, -31.02109455183003, -17.60766189550534, 22.83884746037381,
                28.380477703283688, -16.691276904083473, 1.8105792940720045, -37.87395046635378, -21.93913378183722,
                39.43723994306926, -20.71296414574622, 2.1679624382488223, -49.32438789295014, -29.465880029842584,
                13.085254087036088, -1.5390396304367668, 27.97393922067559, 15.993513682029798, 0.8211060409222076,
                -3.295591393817912, -1.819156241457545, 65.7991061056096, 37.72200796466474, 22.99414125333774], :L) rtol=50eps(elty)
            rhs = elty[1.0 8.0; 2.0 9.0; 3.0 10.0; 4.0 11.0; 5.0 12.0; 6.0 13.0; 7.0 14.0]
            @test rhs ≈ (pm + I) * (chol \ rhs)

            bk = bunchkaufman(pm)
            let P=bk.P, L=bk.L, D=bk.D
                @test P' * L * D * L' * P ≈ Matrix(pm)
            end
            @test inv(bk) ≈ PackedMatrix(n, elty[11.676181149869125, -25.743042670794633, -3.8901359697162454,
                24.102856515354063, 9.176107458680692, -26.29273501144288, 7.584280750861906, 55.837437314564454,
                9.07719922217729, -51.37534555845255, -21.09527940024823, 59.090336295097536, -17.484312579760644,
                1.0285759205945262, -9.226894089251154, -3.294216491388212, 9.912693866880876, -1.9062117598362565,
                49.46819607136643, 19.103438949417765, -55.95001008561188, 16.528078653765096, 9.400381615980521,
                -21.636130300323586, 5.966940819148875, 62.79650629970832, -19.347475787322477, 6.689363759583235], :L)
            @test rhs ≈ pm * (bk \ rhs)
        end
    end
    @testset "Upper scaled" begin
        data = elty[0.9555114416827896,0.19885420667549358,0.5391622763612515,0.9346639709929084,0.3071975474038693,
            0.0019306390378336369,1.0448440948894804,1.1789483273145474,0.3197418723331183,0.5535820925972024,
            0.21248812903556824,0.5814714852042768,0.4999635021104657,0.569274365425051,0.7741030577273389,
            1.392595028004919,0.7556802171356001,0.9119567725517609,0.28805895941204235,0.4682395560638831,
            0.14756643458657948,0.5106476717718449,0.6090548704984261,1.1456513631272307,0.37896523547769884,
            0.3925705265236962,0.13031356174695136,0.45340576959493983]
        pm = PackedMatrix(n, data, :US)
        @test size(pm) == (n, n)
        @test LinearAlgebra.checksquare(pm) == packedside(pm) == packedside(vec(pm)) == n
        @test eltype(pm) === elty
        @test packed_isupper(pm)
        @test !packed_islower(pm)
        @test packed_isscaled(pm)
        @test issymmetric(pm)
        @test ishermitian(pm)
        @test transpose(pm) == pm

        @test collect(pm) == data
        @test [x for x in pm] == data
        @test Matrix(pm) ≈ refmatrix rtol=2eps(elty)
        @test vec(lmul_offdiags!(sqrt(elty(2)), PackedMatrix(Symmetric(refmatrix)))) ≈ vec(pm) rtol=2eps(elty)
        @test [pm[i, j] for i in 1:n, j in 1:n] ≈ refmatrix rtol=2eps(elty)
        for i in 1:n, j in 1:n
            pm[i, j] = refmatrix[i, j]
        end
        @test Matrix(pm) ≈ refmatrix rtol=2eps(elty)
        fill!(pm, zero(elty))
        @test copyto!(pm, refmatrix) === pm
        @test Matrix(pm) ≈ refmatrix
        copy!(pm, refmatrix)
        @test_throws ErrorException PackedMatrix(pm, :L)
        @test diag(pm) == diag(refmatrix)
        @test tr(pm) == tr(refmatrix)

        pmc = copy(pm)
        @test pmc == pm
        @test packed_scale!(pmc) === pmc && pmc == pm
        unscaled = packed_unscale!(pmc)
        @test !packed_isscaled(unscaled)
        @test Matrix(unscaled) ≈ refmatrix rtol=2eps(elty)
        @test packed_scale!(unscaled) ≈ pmc rtol=4eps(elty)
        fill!(pmc, 5.)
        @test all(x -> x == 5. || x == 5 / sqrt(2), pmc)

        @test vec(2 .* pm) == 2data
        @test vec(pm .+ pm) == data .+ data
        @test vec(pm .+ data) == data .+ data
        pmc .= pm .+ 3 .* pm
        @test vec(pmc) == 4data
        @test_throws ErrorException pm .+ unscaled

        if elty ∈ (Float32, Float64) # we need BLAS/LAPACK support for these tests
            copyto!(pmc, pm)
            @test !packed_isscaled(spr!(elty(4), elty.(collect(5:11)), pmc))
            @test vec(pmc) ≈ elty[100.95551144168279,120.14061115800772,144.53916227636125,140.66090723201984,
                168.21722146893316,196.00193063903782,160.73881634477908,192.83364235691266,224.22609164615602,
                256.5535820925972,180.1502517969627,216.41116243025456,252.3535275826881,288.4025377641477,
                324.77410305772736,200.98471338774894,240.5343466059451,280.64485081802036,320.20368844358177,
                360.33109536531254,400.14756643458657,220.361082431507,264.4306668290441,308.8100978477429,
                352.26796888784025,396.2775892813989,440.09214560319185,484.45340576959495] rtol=2eps(elty)
        end

        pms = PackedMatrix(n, sparsevec(data), :US)
        @test dot(pm, pm) ≈ 12.788602219279593 rtol=2eps(elty)
        @test dot(pm, pms) ≈ 12.788602219279593 rtol=2eps(elty)
        @test dot(pms, pms) ≈ 12.788602219279593 rtol=2eps(elty)
        for p in (pm, pms)
            @test norm(pm) ≈ sqrt(12.788602219279593) rtol=2eps(elty)
            @test norm(pm, 1) ≈ 21.571292275978372 rtol=2eps(elty)
            @test norm(pm, Inf) ≈ elty(0.9847133877489483) rtol=2eps(elty)
            @test norm(pm, 0) == 49
            @test norm(pm, -Inf) ≈ elty(0.0019306390378336369) rtol=2eps(elty)
            @test norm(pm, 3.) ≈ 2.0766959423281186 rtol=2eps(elty)
        end

        fill!(pmc, zero(elty))
        pmc[diagind(pmc)] .= elty.(3:9)
        pmcs = PackedMatrix(n, SparseVector{elty,Int}(undef, packedsize(n)), :US)
        pmcs[diagind(pmcs)] .= elty.(3:9)
        @test dot(pm, pmc) ≈ 19.034233988404225 rtol=2eps(elty)
        @test dot(pm, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
        @test dot(pms, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
        for p in (pm, pms)
            @test norm(pmcs) ≈ 16.73320053068151 rtol=2eps(elty)
            @test norm(pmcs, 1) ≈ 42 rtol=2eps(elty)
            @test norm(pmcs, Inf) == 9
            @test norm(pmcs, 0) == n
            @test norm(pmcs, -Inf) == 0
            @test norm(pmcs, 3.) ≈ 12.632719195312756 rtol=2eps(elty)
        end

        pmc .= pm
        @test pmc == pm
        if elty ∈ (Float32, Float64) # we need LAPACK support for these tests
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
                @test ≈(checkvec, refvec, atol=8eps(elty)) || ≈(checkvec, -refvec, atol=8eps(elty))
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
            @test_throws DimensionMismatch eigvals!(pm, 3:5, Vector{elty}(undef, 2))
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
            @test_throws DimensionMismatch eigen!(pm, 3:5, Vector{elty}(undef, 2))
            @test_throws DimensionMismatch eigen!(pm, 3:5, missing, Matrix{elty}(undef, n, 2))
            # spevx total
            @test_throws DimensionMismatch spevx!('V', 'A', pm, nothing, nothing, nothing, nothing, elty(-1),
                Vector{elty}(undef, n -1))
            @test_throws DimensionMismatch spevx!('V', 'A', pm, nothing, nothing, nothing, nothing, elty(-1),
                missing, Matrix{elty}(undef, n, n -1))

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
            @test inv(chol) ≈ PackedMatrix(n, elty[15.493628236589293, 18.055791818905497, 22.83884746037381,
                22.56819523083977, 28.380477703283688, 39.43723994306926, -13.664618497320284, -16.691276904083473,
                -20.71296414574622, 13.085254087036088, 1.6509049892926726, 1.8105792940720054, 2.1679624382488236,
                -1.5390396304367675, 0.8211060409222076, -31.02109455183003, -37.87395046635378, -49.32438789295014,
                27.97393922067559, -3.29559139381791, 65.7991061056096, -17.607661895505345, -21.93913378183722,
                -29.465880029842584, 15.993513682029798, -1.819156241457544, 37.72200796466474, 22.99414125333774]) rtol=150eps(elty)
            rhs = elty[1.0 8.0; 2.0 9.0; 3.0 10.0; 4.0 11.0; 5.0 12.0; 6.0 13.0; 7.0 14.0]
            @test rhs ≈ (pm + I) * (chol \ rhs)

            bk = bunchkaufman(pm)
            let P=bk.P, U=bk.U, D=bk.D
                @test P' * U * D * U' * P ≈ Matrix(pm)
            end
            @test inv(bk) ≈ PackedMatrix(n, elty[11.676181149869125, -25.74304267079464, 55.837437314564454,
                -3.890135969716243, 9.077199222177283, 1.0285759205945262, 24.102856515354063, -51.37534555845253,
                -9.226894089251159,  49.46819607136643, 9.176107458680693, -21.095279400248227, -3.294216491388214,
                19.103438949417768, 9.400381615980521, -26.29273501144287, 59.09033629509751, 9.912693866880879,
                -55.950010085611865, -21.636130300323575, 62.79650629970832, 7.584280750861903, -17.484312579760637,
                -1.9062117598362571, 16.528078653765096, 5.966940819148872, -19.347475787322477, 6.689363759583235])
            @test rhs ≈ pm * (bk \ rhs)
        end
    end
    @testset "Lower scaled" begin
        data = elty[0.9555114416827896,0.19885420667549358,0.9346639709929084,1.0448440948894804,0.21248812903556824,
            1.392595028004919,0.5106476717718449,0.5391622763612515,0.3071975474038693,1.1789483273145474,0.5814714852042768,
            0.7556802171356001,0.6090548704984261,0.0019306390378336369,0.3197418723331183,0.4999635021104657,
            0.9119567725517609,1.1456513631272307,0.5535820925972024,0.569274365425051,0.28805895941204235,
            0.37896523547769884,0.7741030577273389,0.4682395560638831,0.3925705265236962,0.14756643458657948,
            0.13031356174695136,0.45340576959493983]
        pm = PackedMatrix(n, data, :LS)
        @test size(pm) == (n, n)
        @test LinearAlgebra.checksquare(pm) == packedside(pm) == packedside(vec(pm)) == n
        @test eltype(pm) === elty
        @test !packed_isupper(pm)
        @test packed_islower(pm)
        @test packed_isscaled(pm)
        @test issymmetric(pm)
        @test ishermitian(pm)
        @test transpose(pm) == pm

        @test collect(pm) == data
        @test [x for x in pm] == data
        @test Matrix(pm) ≈ refmatrix rtol=2eps(elty)
        @test vec(lmul_offdiags!(sqrt(elty(2)), PackedMatrix(Symmetric(refmatrix, :L)))) ≈ vec(pm) rtol=2eps(elty)
        @test [pm[i, j] for i in 1:n, j in 1:n] ≈ refmatrix rtol=2eps(elty)
        for i in 1:n, j in 1:n
            pm[i, j] = refmatrix[i, j]
        end
        @test Matrix(pm) ≈ refmatrix rtol=2eps(elty)
        fill!(pm, zero(elty))
        @test copyto!(pm, refmatrix) === pm
        @test Matrix(pm) ≈ refmatrix
        copy!(pm, refmatrix)
        @test_throws ErrorException PackedMatrix(pm, :U)
        @test diag(pm) == diag(refmatrix)
        @test tr(pm) == tr(refmatrix)

        pmc = copy(pm)
        @test pmc == pm
        @test packed_scale!(pmc) === pmc && pmc == pm
        unscaled = packed_unscale!(pmc)
        @test !packed_isscaled(unscaled)
        @test Matrix(unscaled) ≈ refmatrix rtol=2eps(elty)
        @test packed_scale!(unscaled) ≈ pmc rtol=4eps(elty)
        fill!(pmc, 5.)
        @test all(x -> x == 5. || x == 5 / sqrt(2), pmc)

        @test vec(2 .* pm) == 2data
        @test vec(pm .+ pm) == data .+ data
        @test vec(pm .+ data) == data .+ data
        pmc .= pm .+ 3 .* pm
        @test vec(pmc) == 4data
        @test_throws ErrorException pm .+ unscaled

        if elty ∈ (Float32, Float64) # we need BLAS/LAPACK support for these tests
            copyto!(pmc, pm)
            @test !packed_isscaled(spr!(elty(4), elty.(collect(5:11)), pmc))
            @test vec(pmc) ≈ elty[100.95551144168279,120.14061115800772,140.66090723201984,160.73881634477908,
                180.1502517969627,200.98471338774894,220.361082431507,144.53916227636125,168.21722146893316,
                192.83364235691266,216.41116243025456,240.5343466059451,264.4306668290441,196.00193063903782,
                224.22609164615602,252.3535275826881,280.64485081802036,308.8100978477429,256.5535820925972,
                288.4025377641477,320.20368844358177,352.26796888784025,324.77410305772736,360.33109536531254,
                396.2775892813989,400.14756643458657,440.09214560319185,484.45340576959495] rtol=2eps(elty)
        end

        pms = PackedMatrix(n, sparsevec(data), :LS)
        @test dot(pm, pm) ≈ 12.788602219279593 rtol=2eps(elty)
        @test dot(pm, pms) ≈ 12.788602219279593 rtol=2eps(elty)
        @test dot(pms, pms) ≈ 12.788602219279593 rtol=2eps(elty)
        for p in (pm, pms)
            @test norm(pm) ≈ sqrt(12.788602219279593) rtol=2eps(elty)
            @test norm(pm, 1) ≈ 21.571292275978372 rtol=2eps(elty)
            @test norm(pm, Inf) ≈ elty(0.9847133877489483) rtol=2eps(elty)
            @test norm(pm, 0) == 49
            @test norm(pm, -Inf) ≈ elty(0.0019306390378336369) rtol=2eps(elty)
            @test norm(pm, 3.) ≈ 2.0766959423281186 rtol=2eps(elty)
        end

        fill!(pmc, zero(elty))
        pmc[diagind(pmc)] .= elty.(3:9)
        pmcs = PackedMatrix(n, SparseVector{elty,Int}(undef, packedsize(n)), :LS)
        pmcs[diagind(pmcs)] .= elty.(3:9)
        @test dot(pm, pmc) ≈ 19.034233988404225 rtol=2eps(elty)
        @test dot(pm, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
        @test dot(pms, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
        for p in (pm, pms)
            @test norm(pmcs) ≈ 16.73320053068151 rtol=2eps(elty)
            @test norm(pmcs, 1) ≈ 42 rtol=2eps(elty)
            @test norm(pmcs, Inf) == 9
            @test norm(pmcs, 0) == n
            @test norm(pmcs, -Inf) == 0
            @test norm(pmcs, 3.) ≈ 12.632719195312756 rtol=2eps(elty)
        end

        pmc .= pm
        @test pmc == pm
        if elty ∈ (Float32, Float64) # we need LAPACK support for these tests
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
                @test ≈(checkvec, refvec, atol=8eps(elty)) || ≈(checkvec, -refvec, atol=8eps(elty))
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
            @test_throws DimensionMismatch eigvals!(pm, 3:5, Vector{elty}(undef, 2))
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
            @test_throws DimensionMismatch eigen!(pm, 3:5, Vector{elty}(undef, 2))
            @test_throws DimensionMismatch eigen!(pm, 3:5, missing, Matrix{elty}(undef, n, 2))
            # spevx total
            @test_throws DimensionMismatch spevx!('V', 'A', pm, nothing, nothing, nothing, nothing, elty(-1),
                Vector{elty}(undef, n -1))
            @test_throws DimensionMismatch spevx!('V', 'A', pm, nothing, nothing, nothing, nothing, elty(-1),
                missing, Matrix{elty}(undef, n, n -1))

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
            @test inv(chol) ≈ PackedMatrix(n, elty[15.493628236589293, 18.05579181890549, 22.568195230839763,
                -13.664618497320282, 1.6509049892926717, -31.02109455183003, -17.60766189550534, 22.83884746037381,
                28.380477703283688, -16.691276904083473, 1.8105792940720045, -37.87395046635378, -21.93913378183722,
                39.43723994306926, -20.71296414574622, 2.1679624382488223, -49.32438789295014, -29.465880029842584,
                13.085254087036088, -1.5390396304367668, 27.97393922067559, 15.993513682029798, 0.8211060409222076,
                -3.295591393817912, -1.819156241457545, 65.7991061056096, 37.72200796466474, 22.99414125333774], :L) rtol=150eps(elty)
            rhs = elty[1.0 8.0; 2.0 9.0; 3.0 10.0; 4.0 11.0; 5.0 12.0; 6.0 13.0; 7.0 14.0]
            @test rhs ≈ (pm + I) * (chol \ rhs)

            bk = bunchkaufman(pm)
            let P=bk.P, L=bk.L, D=bk.D
                @test P' * L * D * L' * P ≈ Matrix(pm)
            end
            @test inv(bk) ≈ PackedMatrix(n, elty[11.676181149869125, -25.743042670794633, -3.8901359697162454,
                24.102856515354063, 9.176107458680692, -26.29273501144288, 7.584280750861906, 55.837437314564454,
                9.07719922217729, -51.37534555845255, -21.09527940024823, 59.090336295097536, -17.484312579760644,
                1.0285759205945262, -9.226894089251154, -3.294216491388212, 9.912693866880876, -1.9062117598362565,
                49.46819607136643, 19.103438949417765, -55.95001008561188, 16.528078653765096, 9.400381615980521,
                -21.636130300323586, 5.966940819148875, 62.79650629970832, -19.347475787322477, 6.689363759583235], :L)
            @test rhs ≈ pm * (bk \ rhs)
        end
    end
end end