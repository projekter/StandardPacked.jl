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
    @testset "Upper unscaled" begin
        data = elty[0.9555114416827896,0.14061115800771273,0.5391622763612515,0.660907232019832,0.21722146893315186,
            0.0019306390378336369,0.7388163447790721,0.8336423569126538,0.22609164615603128,0.5535820925972024,
            0.15025179696269242,0.41116243025455734,0.35352758268808504,0.40253776414772224,0.7741030577273389,
            0.9847133877489483,0.5343466059451054,0.644850818020348,0.2036884435817956,0.33109536531255035,
            0.14756643458657948,0.3610824315069938,0.43066682904413156,0.8100978477428766,0.26796888784023765,
            0.27758928139887895,0.09214560319184117,0.45340576959493983]
        pm = PackedMatrix(7, data)
        @test size(pm) == (7, 7)
        @test LinearAlgebra.checksquare(pm) == 7
        @test eltype(pm) === elty
        @test packed_isupper(pm)
        @test !packed_islower(pm)
        @test !packed_isscaled(pm)

        @test collect(pm) == data
        @test [x for x in pm] == data
        @test Matrix(pm) == refmatrix
        @test [pm[i, j] for i in 1:7, j in 1:7] == refmatrix
        for i in 1:7, j in 1:7
            pm[i, j] = refmatrix[i, j]
        end
        @test Matrix(pm) == refmatrix
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
            tmpout = Vector{elty}(undef, 7)
            @test mul!(tmpout, pm, elty.(collect(1:7)), true, false) === tmpout
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

        pms = PackedMatrix(7, sparsevec(data))
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
        pmcs = PackedMatrix(7, SparseVector{elty,Int}(undef, packedsize(7)))
        pmcs[diagind(pmcs)] .= elty.(3:9)
        @test dot(pm, pmc) ≈ 19.034233988404225 rtol=2eps(elty)
        @test dot(pm, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
        @test dot(pms, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
        for p in (pm, pms)
            @test norm(pmcs) ≈ 16.73320053068151 rtol=2eps(elty)
            @test norm(pmcs, 1) ≈ 42 rtol=2eps(elty)
            @test norm(pmcs, Inf) == 9
            @test norm(pmcs, 0) == 7
            @test norm(pmcs, -Inf) == 0
            @test norm(pmcs, 3.) ≈ 12.632719195312756 rtol=2eps(elty)
        end

        if elty ∈ (Float32, Float64) # we need LAPACK support for these tests
            copyto!(pmc, pm)
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
            pm2 = copy(pm)
            es2 = eigen!(pm2, elty(-.7), elty(1.))
            @test es2.values ≈ @view(es.values[2:6])
            for (checkvec, refvec) in zip(eachcol(es2.vectors), eachcol(@view(es.vectors[:, 2:6])))
                @test ≈(checkvec, refvec, atol=8eps(elty)) || ≈(checkvec, -refvec, atol=8eps(elty))
            end
            copyto!(pm2, pm)
            es2 = eigen!(pm2, 3:5)
            @test es2.values ≈ @view(es.values[3:5]) rtol=4eps(elty)
            for (checkvec, refvec) in zip(eachcol(es2.vectors), eachcol(@view(es.vectors[:, 3:5])))
                @test ≈(checkvec, refvec, atol=8eps(elty)) || ≈(checkvec, -refvec, atol=8eps(elty))
            end
            copyto!(pm2, pm)
            @test eigvals!(pm2, elty(-.7), elty(1.)) ≈ @view(es.values[2:6])
            copyto!(pm2, pm)
            @test eigvals!(pm2, 3:5) ≈ @view(es.values[3:5])
            @test eigmin(pm) ≈ minimum(es.values)
            @test eigmax(pm) ≈ maximum(es.values)

            copyto!(pmc, pm)
            @test !LinearAlgebra.isposdef(pmc)
            @test_throws PosDefException cholesky!(pmc)
            copyto!(pmc, pm)
            @test LinearAlgebra.isposdef(pmc, one(elty))
            @test cholesky!(pmc, shift=one(elty)).U ≈ elty[1.3983960246234932 0.10055174323423234 0.4726180712633075 0.5283312679453536 0.10744581242866912 0.7041734747594656 0.25821185497449645
                0. 1.2365482696982792 0.13723596736343546 0.6312068408830044 0.32377107818793416 0.37486667271940743 0.3272845766073084
                0. 0. 0.8716243956057028 -0.12646679080935472 0.29635897553532703 0.298982588596196 0.737871778986569
                0. 0. 0. 0.9273792313575595 0.19289178031430956 -0.3959064949144887 0.019710925942854545
                0. 0. 0. 0. 1.2380205421376222 0.09840290872367227 -0.06348584599245834
                0. 0. 0. 0. 0. 0.5053369729908973 -0.8290079246701668
                0. 0. 0. 0. 0. 0. 0.20854097637370994]
        end
    end
    @testset "Lower unscaled" begin
        data = elty[0.9555114416827896,0.14061115800771273,0.660907232019832,0.7388163447790721,0.15025179696269242,
            0.9847133877489483,0.3610824315069938,0.5391622763612515,0.21722146893315186,0.8336423569126538,
            0.41116243025455734,0.5343466059451054,0.43066682904413156,0.0019306390378336369,0.22609164615603128,
            0.35352758268808504,0.644850818020348,0.8100978477428766,0.5535820925972024,0.40253776414772224,0.2036884435817956,
            0.26796888784023765,0.7741030577273389,0.33109536531255035,0.27758928139887895,0.14756643458657948,
            0.09214560319184117,0.45340576959493983]
        pm = PackedMatrix(7, data, :L)
        @test size(pm) == (7, 7)
        @test LinearAlgebra.checksquare(pm) == 7
        @test eltype(pm) === elty
        @test !packed_isupper(pm)
        @test packed_islower(pm)
        @test !packed_isscaled(pm)

        @test collect(pm) == data
        @test [x for x in pm] == data
        @test Matrix(pm) == refmatrix
        @test [pm[i, j] for i in 1:7, j in 1:7] == refmatrix
        for i in 1:7, j in 1:7
            pm[i, j] = refmatrix[i, j]
        end
        @test Matrix(pm) == refmatrix
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
            tmpout = Vector{elty}(undef, 7)
            @test mul!(tmpout, pm, elty.(collect(1:7)), true, false) === tmpout
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

        pms = PackedMatrix(7, sparsevec(data), :L)
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
        pmcs = PackedMatrix(7, SparseVector{elty,Int}(undef, packedsize(7)), :L)
        pmcs[diagind(pmcs)] .= elty.(3:9)
        @test dot(pm, pmc) ≈ 19.034233988404225 rtol=2eps(elty)
        @test dot(pm, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
        @test dot(pms, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
        for p in (pm, pms)
            @test norm(pmcs) ≈ 16.73320053068151 rtol=2eps(elty)
            @test norm(pmcs, 1) ≈ 42 rtol=2eps(elty)
            @test norm(pmcs, Inf) == 9
            @test norm(pmcs, 0) == 7
            @test norm(pmcs, -Inf) == 0
            @test norm(pmcs, 3.) ≈ 12.632719195312756 rtol=2eps(elty)
        end

        if elty ∈ (Float32, Float64) # we need LAPACK support for these tests
            copyto!(pmc, pm)
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
            pm2 = copy(pm)
            es2 = eigen!(pm2, elty(-.7), elty(1.))
            @test es2.values ≈ @view(es.values[2:6])
            for (checkvec, refvec) in zip(eachcol(es2.vectors), eachcol(@view(es.vectors[:, 2:6])))
                @test ≈(checkvec, refvec, atol=8eps(elty)) || ≈(checkvec, -refvec, atol=8eps(elty))
            end
            copyto!(pm2, pm)
            es2 = eigen!(pm2, 3:5)
            @test es2.values ≈ @view(es.values[3:5]) rtol=4eps(elty)
            for (checkvec, refvec) in zip(eachcol(es2.vectors), eachcol(@view(es.vectors[:, 3:5])))
                @test ≈(checkvec, refvec, atol=8eps(elty)) || ≈(checkvec, -refvec, atol=8eps(elty))
            end
            copyto!(pm2, pm)
            @test eigvals!(pm2, elty(-.7), elty(1.)) ≈ @view(es.values[2:6])
            copyto!(pm2, pm)
            @test eigvals!(pm2, 3:5) ≈ @view(es.values[3:5])
            @test eigmin(pm) ≈ minimum(es.values)
            @test eigmax(pm) ≈ maximum(es.values)

            copyto!(pmc, pm)
            @test !LinearAlgebra.isposdef(pmc)
            @test_throws PosDefException cholesky!(pmc)
            copyto!(pmc, pm)
            @test LinearAlgebra.isposdef(pmc, one(elty))
            @test cholesky!(pmc, shift=one(elty)).U ≈ elty[1.3983960246234932 0.10055174323423234 0.4726180712633075 0.5283312679453536 0.10744581242866912 0.7041734747594656 0.25821185497449645
                0. 1.2365482696982792 0.13723596736343546 0.6312068408830044 0.32377107818793416 0.37486667271940743 0.3272845766073084
                0. 0. 0.8716243956057028 -0.12646679080935472 0.29635897553532703 0.298982588596196 0.737871778986569
                0. 0. 0. 0.9273792313575595 0.19289178031430956 -0.3959064949144887 0.019710925942854545
                0. 0. 0. 0. 1.2380205421376222 0.09840290872367227 -0.06348584599245834
                0. 0. 0. 0. 0. 0.5053369729908973 -0.8290079246701668
                0. 0. 0. 0. 0. 0. 0.20854097637370994]
        end
    end
    @testset "Upper scaled" begin
        data = elty[0.9555114416827896,0.19885420667549358,0.5391622763612515,0.9346639709929084,0.3071975474038693,
            0.0019306390378336369,1.0448440948894804,1.1789483273145474,0.3197418723331183,0.5535820925972024,
            0.21248812903556824,0.5814714852042768,0.4999635021104657,0.569274365425051,0.7741030577273389,
            1.392595028004919,0.7556802171356001,0.9119567725517609,0.28805895941204235,0.4682395560638831,
            0.14756643458657948,0.5106476717718449,0.6090548704984261,1.1456513631272307,0.37896523547769884,
            0.3925705265236962,0.13031356174695136,0.45340576959493983]
        pm = PackedMatrix(7, data, :US)
        @test size(pm) == (7, 7)
        @test LinearAlgebra.checksquare(pm) == 7
        @test eltype(pm) === elty
        @test packed_isupper(pm)
        @test !packed_islower(pm)
        @test packed_isscaled(pm)

        @test collect(pm) == data
        @test [x for x in pm] == data
        @test Matrix(pm) ≈ refmatrix rtol=2eps(elty)
        @test [pm[i, j] for i in 1:7, j in 1:7] ≈ refmatrix rtol=2eps(elty)
        for i in 1:7, j in 1:7
            pm[i, j] = refmatrix[i, j]
        end
        @test Matrix(pm) ≈ refmatrix rtol=2eps(elty)
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

        pms = PackedMatrix(7, sparsevec(data), :US)
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
        pmcs = PackedMatrix(7, SparseVector{elty,Int}(undef, packedsize(7)), :US)
        pmcs[diagind(pmcs)] .= elty.(3:9)
        @test dot(pm, pmc) ≈ 19.034233988404225 rtol=2eps(elty)
        @test dot(pm, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
        @test dot(pms, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
        for p in (pm, pms)
            @test norm(pmcs) ≈ 16.73320053068151 rtol=2eps(elty)
            @test norm(pmcs, 1) ≈ 42 rtol=2eps(elty)
            @test norm(pmcs, Inf) == 9
            @test norm(pmcs, 0) == 7
            @test norm(pmcs, -Inf) == 0
            @test norm(pmcs, 3.) ≈ 12.632719195312756 rtol=2eps(elty)
        end

        if elty ∈ (Float32, Float64) # we need LAPACK support for these tests
            copyto!(pmc, pm)
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
            pm2 = copy(pm)
            es2 = eigen!(pm2, elty(-.7), elty(1.))
            @test es2.values ≈ @view(es.values[2:6])
            for (checkvec, refvec) in zip(eachcol(es2.vectors), eachcol(@view(es.vectors[:, 2:6])))
                @test ≈(checkvec, refvec, atol=8eps(elty)) || ≈(checkvec, -refvec, atol=8eps(elty))
            end
            copyto!(pm2, pm)
            es2 = eigen!(pm2, 3:5)
            @test es2.values ≈ @view(es.values[3:5]) rtol=4eps(elty)
            for (checkvec, refvec) in zip(eachcol(es2.vectors), eachcol(@view(es.vectors[:, 3:5])))
                @test ≈(checkvec, refvec, atol=8eps(elty)) || ≈(checkvec, -refvec, atol=8eps(elty))
            end
            copyto!(pm2, pm)
            @test eigvals!(pm2, elty(-.7), elty(1.)) ≈ @view(es.values[2:6])
            copyto!(pm2, pm)
            @test eigvals!(pm2, 3:5) ≈ @view(es.values[3:5])
            @test eigmin(pm) ≈ minimum(es.values)
            @test eigmax(pm) ≈ maximum(es.values)

            copyto!(pmc, pm)
            @test !LinearAlgebra.isposdef(pmc)
            @test_throws PosDefException cholesky!(pmc)
            copyto!(pmc, pm)
            @test LinearAlgebra.isposdef(pmc, one(elty))
            @test cholesky!(pmc, shift=one(elty)).U ≈ elty[1.3983960246234932 0.10055174323423234 0.4726180712633075 0.5283312679453536 0.10744581242866912 0.7041734747594656 0.25821185497449645
                0. 1.2365482696982792 0.13723596736343546 0.6312068408830044 0.32377107818793416 0.37486667271940743 0.3272845766073084
                0. 0. 0.8716243956057028 -0.12646679080935472 0.29635897553532703 0.298982588596196 0.737871778986569
                0. 0. 0. 0.9273792313575595 0.19289178031430956 -0.3959064949144887 0.019710925942854545
                0. 0. 0. 0. 1.2380205421376222 0.09840290872367227 -0.06348584599245834
                0. 0. 0. 0. 0. 0.5053369729908973 -0.8290079246701668
                0. 0. 0. 0. 0. 0. 0.20854097637370994]
        end
    end
    @testset "Lower scaled" begin
        data = elty[0.9555114416827896,0.19885420667549358,0.9346639709929084,1.0448440948894804,0.21248812903556824,
            1.392595028004919,0.5106476717718449,0.5391622763612515,0.3071975474038693,1.1789483273145474,0.5814714852042768,
            0.7556802171356001,0.6090548704984261,0.0019306390378336369,0.3197418723331183,0.4999635021104657,
            0.9119567725517609,1.1456513631272307,0.5535820925972024,0.569274365425051,0.28805895941204235,
            0.37896523547769884,0.7741030577273389,0.4682395560638831,0.3925705265236962,0.14756643458657948,
            0.13031356174695136,0.45340576959493983]
        pm = PackedMatrix(7, data, :LS)
        @test size(pm) == (7, 7)
        @test LinearAlgebra.checksquare(pm) == 7
        @test eltype(pm) === elty
        @test !packed_isupper(pm)
        @test packed_islower(pm)
        @test packed_isscaled(pm)

        @test collect(pm) == data
        @test [x for x in pm] == data
        @test Matrix(pm) ≈ refmatrix rtol=2eps(elty)
        @test [pm[i, j] for i in 1:7, j in 1:7] ≈ refmatrix rtol=2eps(elty)
        for i in 1:7, j in 1:7
            pm[i, j] = refmatrix[i, j]
        end
        @test Matrix(pm) ≈ refmatrix rtol=2eps(elty)
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

        pms = PackedMatrix(7, sparsevec(data), :LS)
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
        pmcs = PackedMatrix(7, SparseVector{elty,Int}(undef, packedsize(7)), :LS)
        pmcs[diagind(pmcs)] .= elty.(3:9)
        @test dot(pm, pmc) ≈ 19.034233988404225 rtol=2eps(elty)
        @test dot(pm, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
        @test dot(pms, pmcs) ≈ 19.034233988404225 rtol=2eps(elty)
        for p in (pm, pms)
            @test norm(pmcs) ≈ 16.73320053068151 rtol=2eps(elty)
            @test norm(pmcs, 1) ≈ 42 rtol=2eps(elty)
            @test norm(pmcs, Inf) == 9
            @test norm(pmcs, 0) == 7
            @test norm(pmcs, -Inf) == 0
            @test norm(pmcs, 3.) ≈ 12.632719195312756 rtol=2eps(elty)
        end

        if elty ∈ (Float32, Float64) # we need LAPACK support for these tests
            copyto!(pmc, pm)
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
            pm2 = copy(pm)
            es2 = eigen!(pm2, elty(-.7), elty(1.))
            @test es2.values ≈ @view(es.values[2:6])
            for (checkvec, refvec) in zip(eachcol(es2.vectors), eachcol(@view(es.vectors[:, 2:6])))
                @test ≈(checkvec, refvec, atol=8eps(elty)) || ≈(checkvec, -refvec, atol=8eps(elty))
            end
            copyto!(pm2, pm)
            es2 = eigen!(pm2, 3:5)
            @test es2.values ≈ @view(es.values[3:5]) rtol=4eps(elty)
            for (checkvec, refvec) in zip(eachcol(es2.vectors), eachcol(@view(es.vectors[:, 3:5])))
                @test ≈(checkvec, refvec, atol=8eps(elty)) || ≈(checkvec, -refvec, atol=8eps(elty))
            end
            copyto!(pm2, pm)
            @test eigvals!(pm2, elty(-.7), elty(1.)) ≈ @view(es.values[2:6])
            copyto!(pm2, pm)
            @test eigvals!(pm2, 3:5) ≈ @view(es.values[3:5])
            @test eigmin(pm) ≈ minimum(es.values)
            @test eigmax(pm) ≈ maximum(es.values)

            copyto!(pmc, pm)
            @test !LinearAlgebra.isposdef(pmc)
            @test_throws PosDefException cholesky!(pmc)
            copyto!(pmc, pm)
            @test LinearAlgebra.isposdef(pmc, one(elty))
            @test cholesky!(pmc, shift=one(elty)).U ≈ elty[1.3983960246234932 0.10055174323423234 0.4726180712633075 0.5283312679453536 0.10744581242866912 0.7041734747594656 0.25821185497449645
                0. 1.2365482696982792 0.13723596736343546 0.6312068408830044 0.32377107818793416 0.37486667271940743 0.3272845766073084
                0. 0. 0.8716243956057028 -0.12646679080935472 0.29635897553532703 0.298982588596196 0.737871778986569
                0. 0. 0. 0.9273792313575595 0.19289178031430956 -0.3959064949144887 0.019710925942854545
                0. 0. 0. 0. 1.2380205421376222 0.09840290872367227 -0.06348584599245834
                0. 0. 0. 0. 0. 0.5053369729908973 -0.8290079246701668
                0. 0. 0. 0. 0. 0. 0.20854097637370994]
        end
    end
end end