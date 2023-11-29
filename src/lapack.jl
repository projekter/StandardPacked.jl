using Base: require_one_based_indexing, _realtype
using LinearAlgebra: BlasReal, BlasComplex, BlasFloat, BlasInt, DimensionMismatch, checksquare, chkstride1
using LinearAlgebra.BLAS: libblastrampoline, @blasfunc, chkuplo
import LinearAlgebra.BLAS: spmv!, hpmv!, spr!, vec_pointer_stride
using LinearAlgebra.LAPACK: chklapackerror, chkargsok, chknonsingular

export spmv!, hpmv!, spr!, hpr!,
    gemmt!,
    trttp!, tpttr!, pptrf!, pptrs!, pptri!, spsv!, hpsv!, sptrf!, hptrf!, sptrs!, hptrs!, sptri!, hptri!, spev!, spevx!,
        spevd!, spgv!, spgvx!, spgvd!, hptrd!, opgtr!, opmtr!

# We implement all functions that are missing, if their corresponding dense counterparts are part of Base.
# BLAS Level 2:
# - Hermitian matrix-vector multiply:         sspmv,  dspmv,  chpmv,  zhpmv  (Base: spmv!, hpmv!)
# - symmetric complex matrix-vector multiply:                 cspmv,  zspmv  (not implemented)
# - triangular matrix-vector multiply:        stpmv,  dtpmv,  ctpmv,  ztpmv  (not implemented)
# - rank-1 update:                            sspr,   dspr                   (Base: spr!)
#                                                             chpr,   zhpr   (here: hpr!)
# - symmetric complex rank-1 update:                          cspr,   zspr   (not implemented)
# - rank-2 update:                            sspr2,  dspr2,  chpr2,  zhpr2  (not implemented)
# - triangular solve:                         stpsv,  dtpsv,  ctpsv,  ztpsv  (not implemented)

# BLAS Level 3:
# - matrix-matrix product, triang. update:    sgemmt, dgemmt, cgemmt, zgemmt (here: gemmt!)

# LAPACK
# - conversion full to packed:                strttp, dtrttp, ctrttp, ztrttp (here: trttp!)
# - conversion packed to full:                stpttr, dtpttr, ctpttr, ztpttr (here: tpttr!)
# - Hermitian matrix norm:                    slansp, dlansp, clanhp, zlanhp (not implemented)
# - symmetric complex matrix norm:                            clansp, zlansp (not implemented)
# - triangular banded matrix norm:            slantb, dlantb, clantb, zlantb (not implemented)
# - triangular packed matrix norm:            slantp, dlantp, clantp, zlantp (not implemented)
# - Cholesky driver:                          sppsv,  dppsv,  cppsv,  zppsv  (not implemented)
# - Cholesky expert driver:                   sppsvx, dppsvx, cppsvx, zppsvx (not implemented)
# - Cholesky factorization:                   spptrf, dpptrf, cpptrf, zpptrf (here: pptrf!)
# - Cholesky solver:                          spptrs, dpptrs, cpptrs, zpptrs (here: pptrs!)
# - Cholesky inversion:                       spptri, dpptri, cpptri, zpptri (here: pptri!)
# - Cholesky refinement:                      spprfs, dpprfs, cpprfs, zpprfs (not implemented)
# - Cholesky condition number:                sppcon, dppcon, cppcon, zppcon (not implemented)
# - Cholesky equilibration:                   sppequ, dppequ, cppequ, zppequ (not implemented)
# - symmetric indefinite driver:              sspsv,  dspsv,  chpsv,  zhpsv  (here: spsv!, hpsv!)
# - symmetric indefinite expert driver:       sspsvx, dspsvx, chpsvx, zhpsvx (not implemented)
# - LDLᵀ factorization:                       ssptrf, dsptrf, chptrf, zhptrf (here: sptrf!, hptrf!)
# - LDLᵀ solver:                              ssptrs, dsptrs, chptrs, zhptrs (here: sptrs!, hptrs!)
# - LDLᵀ inversion:                           ssptri, dsptri, chptri, zhptri (here: sptri!, hptri!)
# - LDLᵀ refinement:                          ssprfs, dsprf, csprfs, zsprfs  (not implemented)
# - LDLᵀ condition number:                    sspcon, dspcon, chpcon, zhpcon (not implemented)
# - symmetric complex indef solver:                           cspsv,  zspsv  (here: spsv!)
# - symmetric complex indef expert solver:                    cspsvx, zspsvx (not implemented)
# - LDLᵀ symmetric complex factorization:                     csptrf, zsptrf (here: sptrf!)
# - LDLᵀ symmetric complex solver:                            csptrs, zsptrs (here: sptrs!)
# - LDLᵀ symmetric complex inversion:                         csptri, zsptri (here: sptri!)
# - LDLᵀ symmetric complex refinement:                        csprfs, zsprfs (not implemented)
# - LDLᵀ symmetric complex condition number:                  cspcon, zspcon (not implemented)
# - triangular solver:                        stptrs, dtptrs, ctptrs, ztptrs (not implemented)
# - triangular solve carefully:               slatps, dlatps, clatps, zlatps (not implemented)
# - triangular inversion:                     stptri, dtptri, ctptri, ztptri (not implemented)
# - triangular refinement:                    stprfs, dtprfs, ctprfs, ztprfs (not implemented)
# - triangular condition number:              stpcon, dtpcon, ctpcon, ztpcon (not implemented)
# - eigenvalues:                              sspev,  dspev,  chpev,  zhpev  (here: spev!)
# - eigenvalues, expert:                      sspevx, dspevx, chpevx, zhpevx (here: spevx!)
# - eigenvalues, divide and conquer:          sspevd, dspevd, chpevd, zhpevd (here: spevd!)
# - generalized eigenvalues:                  sspgv,  dspgv,  chpgv,  zhpgv  (here: spgv!)
# - generalized eigenvalues, expert:          sspgvx, dspgvx, chpgvx, zhpgvx (here: spgvx!)
# - generalized eigenvalues, divide&conquer:  sspgvd, dspgvd, chpgvd, zhpgvd (here: spgvd!)
# - reduction to tridiagonal:                 ssptrd, dsptrd, chptrd, zhptrd (here: hptrd!)
# - generate unitary after reduction:         sopgtr, dopgtr, cupgtr, zupgtr (here: opgtr!)
# - multiply by matrix after reduction:       sopmtr, dopmtr, cupmtr, zupmtr (here: opmtr!)
# - reduce generalized to standard form:      sspgst, dspgst, chpgst, zhpgst (not implemented)
# - equilibrate:                              slaqsp, dlaqsp, claqhp, zlaqhp (not implemented)
# - equilibrate symmetrix complex:                            claqsp, zlaqsp (not implemented)

const PtrOrVec{T} = Union{Ptr{T},<:AbstractVector{T}}
const PtrOrMat{T} = Union{Ptr{T},<:AbstractMatrix{T}}
const PtrOrVecOrMat{T} = Union{Ptr{T},<:AbstractVector{T},<:AbstractMatrix{T}}

macro doublefun(name, arg)
    sname = Symbol("s", name)
    sfun = Symbol(sname, "!")
    hname = Symbol("h", name)
    hfun = Symbol(hname, "!")
    esc(quote
        for (f, blasf) in (($(QuoteNode(sfun)), $sname),
                           (isnothing($hname) ? () : (($(QuoteNode(hfun)), $hname),))...)
            @eval $arg
        end
    end)
end

macro doubledoc(name, arg)
    sfun = Symbol("s", name, "!")
    hfun = Symbol("h", name, "!")
    esc(quote
        for (f, T, field, typ, op, prefix) in (($(QuoteNode(sfun)), BlasFloat, "real or complex", "symmetric", "^\\top", "s"),
                                               ($(QuoteNode(hfun)), BlasComplex, "complex", "Hermitian", "'", "h"))
            fn = String(f)
            @eval $arg
        end
    end)
end

macro blascall(fn)
    @assert(fn.head === :call)
    blasname = macroexpand(__module__, :(@blasfunc($(fn.args[1]))))
    esc(:(@ccall(libblastrampoline.$(blasname.value)($(fn.args[2:end]...))::Cvoid)))
end

const warnunscale = """

!!! warning "Scaled matrices"
    The variant of this function that takes a [`SPMatrix`](@ref) also allows for scaled packed matrices. It will
    automatically call [`packed_unscale!`](@ref) on the matrix and return the unscaled result. Do not use the reference to the
    scaled matrix any more, only the result of this function!
"""

for (  hpr,     gemmt,    trttp,    tpttr,    pptrf,    pptrs,    pptri,    spsv,    hpsv,    sptrf,    hptrf,    sptrs,    hptrs,    sptri,    hptri,    spev,    spevx,    spevd,    spgv,    spgvx,    spgvd,    hptrd,    opgtr,    opmtr,  elty) in
   ((nothing, :sgemmt_, :strttp_, :stpttr_, :spptrf_, :spptrs_, :spptri_, :sspsv_, nothing, :ssptrf_, nothing,  :ssptrs_, nothing,  :ssptri_, nothing,  :sspev_, :sspevx_, :sspevd_, :sspgv_, :sspgvx_, :sspgvd_, :ssptrd_, :sopgtr_, :sopmtr_, Float32),
    (nothing, :dgemmt_, :dtrttp_, :dtpttr_, :dpptrf_, :dpptrs_, :dpptri_, :sspsv_, nothing, :dsptrf_, nothing,  :dsptrs_, nothing,  :dsptri_, nothing,  :dspev_, :dspevx_, :dspevd_, :dspgv_, :dspgvx_, :dspgvd_, :dsptrd_, :dopgtr_, :dopmtr_, Float64),
    (:chpr_,  :cgemmt_, :ctrttp_, :ctpttr_, :cpptrf_, :cpptrs_, :cpptri_, :cspsv_, :chpsv_, :csptrf_, :chptrf_, :csptrs_, :chptrs_, :csptri_, :chptri_, :chpev_, :chpevx_, :chpevd_, :chpgv_, :chpgvx_, :chpgvd_, :chptrd_, :cupgtr_, :cupmtr_, ComplexF32),
    (:zhpr_,  :zgemmt_, :ztrttp_, :ztpttr_, :zpptrf_, :zpptrs_, :zpptri_, :zspsv_, :zhpsv_, :zsptrf_, :zhptrf_, :zsptrs_, :zhptrs_, :zsptri_, :zhptri_, :zhpev_, :zhpevx_, :zhpevd_, :zhpgv_, :zhpgvx_, :zhpgvd_, :zhptrd_, :zupgtr_, :zupmtr_, ComplexF64))
    relty = _realtype(elty)
    isnothing(hpr) || @eval begin
        # SUBROUTINE ZHPR(UPLO,N,ALPHA,X,INCX,AP)
        # *       .. Scalar Arguments ..
        # *       DOUBLE PRECISION ALPHA
        # *       INTEGER INCX,N
        # *       CHARACTER UPLO
        # *       ..
        # *       .. Array Arguments ..
        # *       COMPLEX*16 AP(*),X(*)
        function hpr!(uplo::AbstractChar, n::Integer, α::$elty, x::PtrOrVec{$elty}, incx::Integer, AP::PtrOrVec{$elty})
            chkuplo(uplo)
            @blascall $hpr(
                uplo::Ref{UInt8}, n::Ref{BlasInt}, α::Ref{$elty}, x::Ptr{$elty}, incx::Ref{BlasInt}, AP::Ptr{$elty}, 1::Clong
            )
            return AP
        end
    end
    if BLAS.lbt_get_forward(@blasfunc(gemmt),
        BLAS.USE_BLAS64 ? BLAS.LBT_INTERFACE_ILP64 : BLAS.LBT_INTERFACE_LP64) ∈ (BLAS.lbt_get_default_func(), Ptr{Cvoid}(-1))
        @eval begin
            # Not part of the reference BLAS (yet), but present in many implementations, also Julia 1.9
            # Requires at least OpenBlas 0.3.22, but Julia currently maximally ships with 0.3.17. But with this construction, we
            # make the function available anyway - just load MKL before this package. When the function is not found, the more
            # inefficient gemm! is called.
            function gemmt!(uplo::AbstractChar, transA::AbstractChar, transB::AbstractChar, alpha::Union{$elty,Bool},
                A::AbstractVecOrMat{$elty}, B::AbstractVecOrMat{$elty}, beta::Union{$elty, Bool}, C::AbstractVecOrMat{$elty})
                return BLAS.gemm!(transA, transB, alpha, A, B, beta, C)
            end
        end
    else
        @eval begin
            # SUBROUTINE DGEMMT( UPLO, TRANSA, TRANSB, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
            #       .. Scalar Arguments ..
            #       DOUBLE PRECISION ALPHA,BETA
            #       INTEGER K,LDA,LDB,LDC,N
            #       CHARACTER TRANSA,TRANSB, UPLO
            #       ..
            #       .. Array Arguments ..
            #       DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
            function gemmt!(uplo::AbstractChar, transA::AbstractChar, transB::AbstractChar, alpha::Union{$elty,Bool},
                A::AbstractVecOrMat{$elty}, B::AbstractVecOrMat{$elty}, beta::Union{$elty,Bool}, C::AbstractVecOrMat{$elty})
                require_one_based_indexing(A, B, C)
                chkuplo(uplo)
                m = size(A, transA == 'N' ? 1 : 2)
                ka = size(A, transA == 'N' ? 2 : 1)
                kb = size(B, transB == 'N' ? 1 : 2)
                n = size(B, transB == 'N' ? 2 : 1)
                if ka != kb || m != checksquare(C) || n != m
                    throw(DimensionMismatch("A has size ($m,$ka), B has size ($kb,$n), C has size $(size(C))"))
                end
                chkstride1(A, B, C)
                @blascall $gemmt(
                    uplo::Ref{UInt8}, transA::Ref{UInt8}, transB::Ref{UInt8}, alpha::Ref{$elty}, A::Ptr{$elty},
                    max(1, stride(A, 2))::Ref{BlasInt}, B::Ptr{$elty}, max(1, stride(B, 2))::Ref{BlasInt}, beta::Ref{$elty},
                    C::Ptr{$elty}, max(1, stride(C, 2))::Ref{BlasInt}, 1::Clong, 1::Clong, 1::Clong
                )
                return C
            end
        end
    end
    @eval begin
        # SUBROUTINE DTRTTP( UPLO, N, A, LDA, AP, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          UPLO
        #       INTEGER            INFO, N, LDA
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   A( LDA, * ), AP( * )
        function trttp!(uplo::AbstractChar, n::Integer, A::PtrOrMat{$elty}, lda::Integer, AP::PtrOrVec{$elty})
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $trttp(
                uplo::Ref{UInt8}, n::Ref{BlasInt}, A::Ptr{$elty}, lda::Ref{BlasInt}, AP::Ptr{$elty}, info::Ref{BlasInt}
            )
            chklapackerror(info[])
            return AP
        end

        # SUBROUTINE DTPTTR( UPLO, N, AP, A, LDA, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          UPLO
        #       INTEGER            INFO, N, LDA
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   A( LDA, * ), AP( * )
        function tpttr!(uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty}, A::PtrOrMat{$elty}, lda::Integer)
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $tpttr(
                uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty}, A::Ptr{$elty}, lda::Ref{BlasInt}, info::Ref{BlasInt},
                1::Clong
            )
            chklapackerror(info[])
            return A
        end

        # SUBROUTINE DPPTRF( UPLO, N, AP, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          UPLO
        #       INTEGER            INFO, N
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   AP( * )
        function pptrf!(uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty})
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $pptrf(
                uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty}, info::Ref{BlasInt}, 1::Clong
            )
            chkargsok(info[])
            #info[] > 0 means the leading minor of order info[] is not positive definite
            #ordinarily, throw Exception here, but return error code here
            #this simplifies isposdef! and factorize
            return AP, info[] # info stored in Cholesky
        end

        # SUBROUTINE DPPTRS( UPLO, N, NRHS, AP, B, LDB, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          UPLO
        #       INTEGER            INFO, LDB, N, NRHS
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   AP( * ), B( LDB, * )
        function pptrs!(uplo::AbstractChar, n::Integer, nrhs::Integer, AP::PtrOrVec{$elty}, B::PtrOrVecOrMat{$elty},
            ldb::Integer)
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $pptrs(
                uplo::Ref{UInt8}, n::Ref{BlasInt}, nrhs::Ref{BlasInt}, AP::Ptr{$elty}, B::Ptr{$elty}, ldb::Ref{BlasInt},
                info::Ref{BlasInt}, 1::Clong
            )
            chklapackerror(info[])
            return B
        end

        # SUBROUTINE DPPTRI( UPLO, N, AP, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          UPLO
        #       INTEGER            INFO, N
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   AP( * )
        function pptri!(uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty})
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $pptri(
                uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty}, info::Ref{BlasInt}, 1::Clong
            )
            chkargsok(info[])
            chknonsingular(info[])
            return AP
        end
    end
    @doublefun psv begin
        # SUBROUTINE DSPSV( UPLO, N, NRHS, AP, IPIV, B, LDB, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          UPLO
        #       INTEGER            INFO, LDB, N, NRHS
        # *     ..
        # *     .. Array Arguments ..
        #       INTEGER            IPIV( * )
        #       DOUBLE PRECISION   AP( * ), B( LDB, * )
        function $f(uplo::AbstractChar, n::Integer, nrhs::Integer, AP::PtrOrVec{$elty}, ipiv::PtrOrVec{BlasInt},
            B::PtrOrMat{$elty}, ldb::Integer)
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $blasf(
                uplo::Ref{UInt8}, n::Ref{BlasInt}, nrhs::Ref{BlasInt}, AP::Ptr{$elty}, ipiv::Ptr{BlasInt}, B::Ptr{$elty},
                ldb::Ref{BlasInt}, info::Ref{BlasInt}, 1::Clong
            )
            chklapackerror(info[])
            return B, AP, ipiv
        end
    end
    @doublefun ptrf begin
        # SUBROUTINE DSPTRF( UPLO, N, AP, IPIV, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          UPLO
        #       INTEGER            INFO, N
        # *     ..
        # *     .. Array Arguments ..
        #       INTEGER            IPIV( * )
        #       DOUBLE PRECISION   AP( * )
        function $f(uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty}, ipiv::PtrOrVec{BlasInt})
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $blasf(
                uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty}, ipiv::Ptr{BlasInt}, info::Ref{BlasInt}, 1::Clong
            )
            chkargsok(info[])
            return AP, ipiv, info[]
        end
    end
    @doublefun ptrs begin
        # SUBROUTINE DSPTRS( UPLO, N, NRHS, AP, IPIV, B, LDB, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          UPLO
        #       INTEGER            INFO, LDB, N, NRHS
        # *     ..
        # *     .. Array Arguments ..
        #       INTEGER            IPIV( * )
        #       DOUBLE PRECISION   AP( * ), B( LDB, * )
        function $f(uplo::AbstractChar, n::Integer, nrhs::Integer, AP::PtrOrVec{$elty}, ipiv::PtrOrVec{BlasInt},
            B::PtrOrMat{$elty}, ldb::Integer)
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $blasf(
                uplo::Ref{UInt8}, n::Ref{BlasInt}, nrhs::Ref{BlasInt}, AP::Ptr{$elty}, ipiv::Ptr{BlasInt}, B::Ptr{$elty},
                ldb::Ref{BlasInt}, info::Ref{BlasInt}, 1::Clong
            )
            chklapackerror(info[])
            return B
        end
    end
    @doublefun ptri begin
        # SUBROUTINE DSPTRI( UPLO, N, AP, IPIV, WORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          UPLO
        #       INTEGER            INFO, N
        # *     ..
        # *     .. Array Arguments ..
        #       INTEGER            IPIV( * )
        #       DOUBLE PRECISION   AP( * ), WORK( * )
        function $f(uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty}, ipiv::PtrOrVec{BlasInt}, work::PtrOrVec{$elty})
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $blasf(
                uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty}, ipiv::Ptr{BlasInt}, work::Ptr{$elty}, info::Ref{BlasInt},
                1::Clong
            )
            chklapackerror(info[])
            return AP
        end
    end
    elty <: Real && @eval begin
        # SUBROUTINE DSPEV( JOBZ, UPLO, N, AP, W, Z, LDZ, WORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBZ, UPLO
        #       INTEGER            INFO, LDZ, N
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   AP( * ), W( * ), WORK( * ), Z( LDZ, * )
        function spev!(jobz::AbstractChar, uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty}, W::PtrOrVec{$elty},
            Z::PtrOrMat{$elty}, ldz::Integer, work::PtrOrVec{$elty})
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $spev(
                jobz::Ref{UInt8}, uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty}, W::Ptr{$elty}, Z::Ptr{$elty},
                ldz::Ref{BlasInt}, work::Ptr{$elty}, info::Ref{BlasInt}, 1::Clong, 1::Clong
            )
            chklapackerror(info[])
            return W, Z
        end

        # SUBROUTINE DSPEVX( JOBZ, RANGE, UPLO, N, AP, VL, VU, IL, IU, ABSTOL, M, W, Z, LDZ, WORK, IWORK, IFAIL, INFO )
        #    *     .. Scalar Arguments ..
        #          CHARACTER          JOBZ, RANGE, UPLO
        #          INTEGER            IL, INFO, IU, LDZ, M, N
        #          DOUBLE PRECISION   ABSTOL, VL, VU
        #    *     ..
        #    *     .. Array Arguments ..
        #          INTEGER            IFAIL( * ), IWORK( * )
        #          DOUBLE PRECISION   AP( * ), W( * ), WORK( * ), Z( LDZ, * )
        function spevx!(jobz::AbstractChar, range::AbstractChar, uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty},
                vl::Union{Nothing,$elty}, vu::Union{Nothing,$elty}, il::Union{Nothing,<:Integer}, iu::Union{Nothing,<:Integer},
                abstol::$elty, W::PtrOrVec{$elty}, Z::PtrOrMat{$elty}, ldz::Integer, work::PtrOrVec{$elty},
                iwork::PtrOrVec{BlasInt}, ifail::PtrOrVec{BlasInt})
            chkuplo(uplo)
            m = Ref{BlasInt}()
            info = Ref{BlasInt}()
            @blascall $spevx(
                jobz::Ref{UInt8}, range::Ref{UInt8}, uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty},
                (isnothing(vl) ? C_NULL : Ref(vl))::Ptr{$elty},
                (isnothing(vu) ? C_NULL : Ref(vu))::Ptr{$elty},
                (isnothing(il) ? C_NULL : Ref(BlasInt(il)))::Ptr{BlasInt},
                (isnothing(iu) ? C_NULL : Ref(BlasInt(iu)))::Ptr{BlasInt},
                abstol::Ref{$elty}, m::Ref{BlasInt}, W::Ptr{$elty}, Z::Ptr{$elty}, ldz::Ref{BlasInt}, work::Ptr{$elty},
                iwork::Ptr{BlasInt}, ifail::Ptr{BlasInt}, info::Ref{BlasInt}, 1::Clong, 1::Clong, 1::Clong
            )
            chkargsok(info[])
            return m[], W, Z, info[], ifail
        end

        # SUBROUTINE DSPEVD( JOBZ, UPLO, N, AP, W, Z, LDZ, WORK, LWORK, IWORK, LIWORK, INFO )
        #    *     .. Scalar Arguments ..
        #          CHARACTER          JOBZ, UPLO
        #          INTEGER            INFO, LDZ, LIWORK, LWORK, N
        #    *     ..
        #    *     .. Array Arguments ..
        #          INTEGER            IWORK( * )
        #          DOUBLE PRECISION   AP( * ), W( * ), WORK( * ), Z( LDZ, * )
        function spevd!(jobz::AbstractChar, uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty}, W::PtrOrVec{$elty},
            Z::PtrOrMat{$elty}, ldz::Integer, work::PtrOrVec{$elty}, lwork::Integer, iwork::PtrOrVec{BlasInt}, liwork::Integer)
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $spevd(
                jobz::Ref{UInt8}, uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty}, W::Ptr{$elty}, Z::Ptr{$elty},
                ldz::Ref{BlasInt}, work::Ptr{$elty}, lwork::Ref{BlasInt}, iwork::Ptr{BlasInt}, liwork::Ref{BlasInt},
                info::Ref{BlasInt}, 1::Clong, 1::Clong
            )
            chklapackerror(info[])
            return W, Z
        end

        # SUBROUTINE DSPGV( ITYPE, JOBZ, UPLO, N, AP, BP, W, Z, LDZ, WORK, INFO )
        #    *     .. Scalar Arguments ..
        #          CHARACTER          JOBZ, UPLO
        #          INTEGER            INFO, ITYPE, LDZ, N
        #    *     ..
        #    *     .. Array Arguments ..
        #          DOUBLE PRECISION   AP( * ), BP( * ), W( * ), WORK( * ), Z( LDZ, * )
        function spgv!(itype::Integer, jobz::AbstractChar, uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty},
            BP::PtrOrVec{$elty}, W::PtrOrVec{$elty}, Z::PtrOrMat{$elty}, ldz::Integer, work::PtrOrVec{$elty})
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $spgv(
                itype::Ref{BlasInt}, jobz::Ref{UInt8}, uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty}, BP::Ptr{$elty},
                W::Ptr{$elty}, Z::Ptr{$elty}, ldz::Ref{BlasInt}, work::Ptr{$elty}, info::Ref{BlasInt}, 1::Clong, 1::Clong
            )
            chklapackerror(info[])
            return W, Z
        end

        # SUBROUTINE DSPGVX( ITYPE, JOBZ, RANGE, UPLO, N, AP, BP, VL, VU, IL, IU, ABSTOL, M, W, Z, LDZ, WORK, IWORK, IFAIL,
        #                    INFO )
        #    *     .. Scalar Arguments ..
        #          CHARACTER          JOBZ, RANGE, UPLO
        #          INTEGER            IL, INFO, ITYPE, IU, LDZ, M, N
        #          DOUBLE PRECISION   ABSTOL, VL, VU
        #    *     ..
        #    *     .. Array Arguments ..
        #          INTEGER            IFAIL( * ), IWORK( * )
        #          DOUBLE PRECISION   AP( * ), BP( * ), W( * ), WORK( * ), Z( LDZ, * )
        function spgvx!(itype::Integer, jobz::AbstractChar, range::AbstractChar, uplo::AbstractChar, n::Integer,
            AP::PtrOrVec{$elty}, BP::PtrOrVec{$elty}, vl::Union{Nothing,$elty}, vu::Union{Nothing,$elty},
            il::Union{Nothing,<:Integer}, iu::Union{Nothing,<:Integer}, abstol::$elty, W::PtrOrVec{$elty}, Z::PtrOrMat{$elty},
            ldz::Integer, work::PtrOrVec{$elty}, iwork::PtrOrVec{BlasInt}, ifail::PtrOrVec{BlasInt})
            chkuplo(uplo)
            m = Ref{BlasInt}()
            info = Ref{BlasInt}()
            @blascall $spgvx(
                itype::Ref{BlasInt}, jobz::Ref{UInt8}, range::Ref{UInt8}, uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty},
                BP::Ptr{$elty}, (isnothing(vl) ? C_NULL : Ref(vl))::Ptr{$elty},
                (isnothing(vu) ? C_NULL : Ref(vu))::Ptr{$elty},
                (isnothing(il) ? C_NULL : Ref(BlasInt(il)))::Ptr{BlasInt},
                (isnothing(iu) ? C_NULL : Ref(BlasInt(iu)))::Ptr{BlasInt},
                abstol::Ref{$elty}, m::Ref{BlasInt}, W::Ptr{$elty}, Z::Ptr{$elty}, ldz::Ref{BlasInt}, work::Ptr{$elty},
                iwork::Ptr{BlasInt}, ifail::Ptr{BlasInt}, info::Ref{BlasInt}, 1::Clong, 1::Clong, 1::Clong
            )
            chkargsok(info[])
            return m[], W, Z, info[], ifail
        end

        # SUBROUTINE DSPGVD( ITYPE, JOBZ, UPLO, N, AP, BP, W, Z, LDZ, WORK, LWORK, IWORK, LIWORK, INFO )
        #    *     .. Scalar Arguments ..
        #          CHARACTER          JOBZ, UPLO
        #          INTEGER            INFO, ITYPE, LDZ, LIWORK, LWORK, N
        #    *     ..
        #    *     .. Array Arguments ..
        #          INTEGER            IWORK( * )
        #          DOUBLE PRECISION   AP( * ), BP( * ), W( * ), WORK( * ), Z( LDZ, * )
        function spgvd!(itype::Integer, jobz::AbstractChar, uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty},
            BP::PtrOrVec{$elty}, W::PtrOrVec{$elty}, Z::PtrOrMat{$elty}, ldz::Integer, work::PtrOrVec{$elty}, lwork::Integer,
            iwork::PtrOrVec{BlasInt}, liwork::Integer)
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $spgvd(
                itype::Ref{BlasInt}, jobz::Ref{UInt8}, uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty}, BP::Ptr{$elty},
                W::Ptr{$elty}, Z::Ptr{$elty}, ldz::Ref{BlasInt}, work::Ptr{$elty}, lwork::Ref{BlasInt}, iwork::Ptr{BlasInt},
                liwork::Ref{BlasInt}, info::Ref{BlasInt}, 1::Clong, 1::Clong
            )
            chklapackerror(info[])
            return W, Z
        end
    end
    elty <: Complex && @eval begin
        # SUBROUTINE ZHPEV( JOBZ, UPLO, N, AP, W, Z, LDZ, WORK, RWORK, INFO )
        #    *     .. Scalar Arguments ..
        #          CHARACTER          JOBZ, UPLO
        #          INTEGER            INFO, LDZ, N
        #    *     ..
        #    *     .. Array Arguments ..
        #          DOUBLE PRECISION   RWORK( * ), W( * )
        #          COMPLEX*16         AP( * ), WORK( * ), Z( LDZ, * )
        function spev!(jobz::AbstractChar, uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty}, W::PtrOrVec{$relty},
            Z::PtrOrMat{$elty}, ldz::Integer, work::PtrOrVec{$elty}, rwork::PtrOrVec{$relty})
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $spev(
                jobz::Ref{UInt8}, uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty}, W::Ptr{$elty}, Z::Ptr{$elty},
                ldz::Ref{BlasInt}, work::Ptr{$elty}, rwork::Ptr{$relty}, info::Ref{BlasInt}, 1::Clong, 1::Clong
            )
            chklapackerror(info[])
            return W, Z
        end

        # SUBROUTINE ZHPEVX( JOBZ, RANGE, UPLO, N, AP, VL, VU, IL, IU, ABSTOL, M, W, Z, LDZ, WORK, RWORK, IWORK, IFAIL, INFO )
        #    *     .. Scalar Arguments ..
        #          CHARACTER          JOBZ, RANGE, UPLO
        #          INTEGER            IL, INFO, IU, LDZ, M, N
        #          DOUBLE PRECISION   ABSTOL, VL, VU
        #    *     ..
        #    *     .. Array Arguments ..
        #          INTEGER            IFAIL( * ), IWORK( * )
        #          DOUBLE PRECISION   RWORK( * ), W( * )
        #          COMPLEX*16         AP( * ), WORK( * ), Z( LDZ, * )
        function spevx!(jobz::AbstractChar, range::AbstractChar, uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty},
            vl::Union{Nothing,$relty}, vu::Union{Nothing,$relty}, il::Union{Nothing,<:Integer}, iu::Union{Nothing,<:Integer},
            abstol::$relty, W::PtrOrVec{$relty}, Z::PtrOrMat{$elty}, ldz::Integer, work::PtrOrVec{$elty},
            rwork::PtrOrVec{$relty}, iwork::PtrOrVec{BlasInt}, ifail::PtrOrVec{BlasInt})
            chkuplo(uplo)
            m = Ref{BlasInt}()
            info = Ref{BlasInt}()
            @blascall $spevx(
                jobz::Ref{UInt8}, range::Ref{UInt8}, uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty},
                (isnothing(vl) ? C_NULL : Ref(vl))::Ptr{$relty}, (isnothing(vu) ? C_NULL : Ref(vu))::Ptr{$relty},
                (isnothing(il) ? C_NULL : Ref(BlasInt(il)))::Ptr{BlasInt},
                (isnothing(iu) ? C_NULL : Ref(BlasInt(iu)))::Ptr{BlasInt},
                abstol::Ref{$elty}, m::Ref{BlasInt}, W::Ptr{$relty}, Z::Ptr{$elty}, ldz::Ref{BlasInt},
                work::Ptr{$elty}, rwork::Ptr{$relty}, iwork::Ptr{BlasInt}, ifail::Ptr{BlasInt}, info::Ref{BlasInt}, 1::Clong,
                1::Clong, 1::Clong
            )
            chkargsok(info[])
            return m[], W, Z, info[], ifail
        end

        # SUBROUTINE ZHPEVD( JOBZ, UPLO, N, AP, W, Z, LDZ, WORK, LWORK, RWORK, LRWORK, IWORK, LIWORK, INFO )
        #    *     .. Scalar Arguments ..
        #          CHARACTER          JOBZ, UPLO
        #          INTEGER            INFO, LDZ, LIWORK, LRWORK, LWORK, N
        #    *     ..
        #    *     .. Array Arguments ..
        #          INTEGER            IWORK( * )
        #          DOUBLE PRECISION   RWORK( * ), W( * )
        #          COMPLEX*16         AP( * ), WORK( * ), Z( LDZ, * )
        function spevd!(jobz::AbstractChar, uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty},
            W::PtrOrVec{$relty}, Z::PtrOrMat{$elty}, ldz::Integer, work::PtrOrVec{$elty}, lwork::Integer,
            rwork::PtrOrVec{$relty}, lrwork::Integer, iwork::PtrOrVec{BlasInt}, liwork::Integer)
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $spevd(
                jobz::Ref{UInt8}, uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty}, W::Ptr{$relty}, Z::Ptr{$elty},
                ldz::Ref{BlasInt}, work::Ptr{$elty}, lwork::Ref{BlasInt}, rwork::Ptr{$relty}, lrwork::Ref{BlasInt},
                iwork::Ptr{BlasInt}, liwork::Ref{BlasInt}, info::Ref{BlasInt}, 1::Clong, 1::Clong
            )
            chklapackerror(info[])
            return W, Z
        end

        # SUBROUTINE ZHPGV( ITYPE, JOBZ, UPLO, N, AP, BP, W, Z, LDZ, WORK, RWORK, INFO )
        #    *     .. Scalar Arguments ..
        #          CHARACTER          JOBZ, UPLO
        #          INTEGER            INFO, ITYPE, LDZ, N
        #    *     ..
        #    *     .. Array Arguments ..
        #          DOUBLE PRECISION   RWORK( * ), W( * )
        #          COMPLEX*16         AP( * ), BP( * ), WORK( * ), Z( LDZ, * )
        function spgv!(itype::Integer, jobz::AbstractChar, uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty},
            BP::PtrOrVec{$elty}, W::PtrOrVec{$relty}, Z::PtrOrMat{$elty}, ldz::Integer, work::PtrOrVec{$elty},
            rwork::PtrOrVec{$relty})
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $spgv(
                itype::Ref{BlasInt}, jobz::Ref{UInt8}, uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty}, BP::Ptr{$elty},
                W::Ptr{$elty}, Z::Ptr{$elty}, ldz::Ref{BlasInt}, work::Ptr{$elty}, rwork::Ptr{$relty},
                info::Ref{BlasInt}, 1::Clong, 1::Clong
            )
            chklapackerror(info[])
            return W, Z
        end

        # SUBROUTINE ZHPGVX( ITYPE, JOBZ, RANGE, UPLO, N, AP, BP, VL, VU, IL, IU, ABSTOL, M, W, Z, LDZ, WORK, RWORK, IWORK,
        #                    IFAIL, INFO )
        #    *     .. Scalar Arguments ..
        #          CHARACTER          JOBZ, RANGE, UPLO
        #          INTEGER            IL, INFO, ITYPE, IU, LDZ, M, N
        #          DOUBLE PRECISION   ABSTOL, VL, VU
        #    *     ..
        #    *     .. Array Arguments ..
        #          INTEGER            IFAIL( * ), IWORK( * )
        #          DOUBLE PRECISION   RWORK( * ), W( * )
        #          COMPLEX*16         AP( * ), BP( * ), WORK( * ), Z( LDZ, * )
        function spgvx!(itype::Integer, jobz::AbstractChar, range::AbstractChar, uplo::AbstractChar, n::Integer,
            AP::PtrOrVec{$elty}, BP::PtrOrVec{$elty}, vl::Union{Nothing,$relty}, vu::Union{Nothing,$relty},
            il::Union{Nothing,<:Integer}, iu::Union{Nothing,<:Integer}, abstol::$relty, W::PtrOrVec{$relty},
            Z::PtrOrMat{$elty}, ldz::Integer, work::PtrOrVec{$elty}, rwork::PtrOrVec{$relty}, iwork::PtrOrVec{BlasInt},
            ifail::PtrOrVec{BlasInt})
            chkuplo(uplo)
            m = Ref{BlasInt}()
            info = Ref{BlasInt}()
            @blascall $spgvx(
                itype::Ref{BlasInt}, jobz::Ref{UInt8}, range::Ref{UInt8}, uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty},
                BP::Ptr{$elty}, (isnothing(vl) ? C_NULL : Ref(vl))::Ptr{$relty},
                (isnothing(vu) ? C_NULL : Ref(vu))::Ptr{$relty},
                (isnothing(il) ? C_NULL : Ref(BlasInt(il)))::Ptr{BlasInt},
                (isnothing(iu) ? C_NULL : Ref(BlasInt(iu)))::Ptr{BlasInt},
                abstol::Ref{$elty}, m::Ref{BlasInt}, W::Ptr{$relty}, Z::Ptr{$elty}, ldz::Ref{BlasInt},
                work::Ptr{$elty}, rwork::Ptr{$relty}, iwork::Ptr{BlasInt}, ifail::Ptr{BlasInt}, info::Ref{BlasInt}, 1::Clong,
                1::Clong, 1::Clong
            )
            chkargsok(info[])
            return m[], W, Z, info[], ifail
        end

        # SUBROUTINE ZHPGVD( ITYPE, JOBZ, UPLO, N, AP, BP, W, Z, LDZ, WORK, LWORK, RWORK, LRWORK, IWORK, LIWORK, INFO )
        #    *     .. Scalar Arguments ..
        #          CHARACTER          JOBZ, UPLO
        #          INTEGER            INFO, ITYPE, LDZ, LIWORK, LRWORK, LWORK, N
        #    *     ..
        #    *     .. Array Arguments ..
        #          INTEGER            IWORK( * )
        #          DOUBLE PRECISION   RWORK( * ), W( * )
        #          COMPLEX*16         AP( * ), BP( * ), WORK( * ), Z( LDZ, * )
        function spgvd!(itype::Integer, jobz::AbstractChar, uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty},
            BP::PtrOrVec{$elty}, W::PtrOrVec{$relty}, Z::PtrOrMat{$elty}, ldz::Integer, work::PtrOrVec{$elty},
            lwork::Integer, rwork::PtrOrVec{$relty}, lrwork::Integer, iwork::PtrOrVec{BlasInt}, liwork::Integer)
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $spgvd(
                itype::Ref{BlasInt}, jobz::Ref{UInt8}, uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty}, BP::Ptr{$elty},
                W::Ptr{$relty}, Z::Ptr{$elty}, ldz::Ref{BlasInt}, work::Ptr{$elty}, lwork::Ref{BlasInt}, rwork::Ptr{$relty},
                lrwork::Ref{BlasInt}, iwork::Ptr{BlasInt}, liwork::Ref{BlasInt}, info::Ref{BlasInt}, 1::Clong, 1::Clong
            )
            chklapackerror(info[])
            return W, Z
        end
    end
    @eval begin
        # SUBROUTINE ZHPTRD( UPLO, N, AP, D, E, TAU, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          UPLO
        #       INTEGER            INFO, N
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   D( * ), E( * )
        #       COMPLEX*16         AP( * ), TAU( * )
        function hptrd!(uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty}, D::PtrOrVec{$relty}, E::PtrOrVec{$relty},
            τ::PtrOrVec{$elty})
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $hptrd(
                uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty}, D::Ptr{$relty}, E::Ptr{$relty},
                τ::Ptr{$elty}, info::Ref{BlasInt}, 1::Clong
            )
            chklapackerror(info[])
            return AP, τ, D, E
        end

        # SUBROUTINE DOPGTR( UPLO, N, AP, TAU, Q, LDQ, WORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          UPLO
        #       INTEGER            INFO, LDQ, N
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   AP( * ), Q( LDQ, * ), TAU( * ), WORK( * )
        function opgtr!(uplo::AbstractChar, n::Integer, AP::PtrOrVec{$elty}, τ::PtrOrVec{$elty}, Q::PtrOrMat{$elty},
            ldq::Integer, work::PtrOrVec{$elty})
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $opgtr(
                uplo::Ref{UInt8}, n::Ref{BlasInt}, AP::Ptr{$elty}, τ::Ptr{$elty}, Q::Ptr{$elty}, ldq::Ref{BlasInt},
                work::Ptr{$elty}, info::Ref{BlasInt}, 1::Clong
            )
            chklapackerror(info[])
            return Q
        end

        # SUBROUTINE DOPMTR( SIDE, UPLO, TRANS, M, N, AP, TAU, C, LDC, WORK, INFO )
        #    *     .. Scalar Arguments ..
        #          CHARACTER          SIDE, TRANS, UPLO
        #          INTEGER            INFO, LDC, M, N
        #    *     ..
        #    *     .. Array Arguments ..
        #          DOUBLE PRECISION   AP( * ), C( LDC, * ), TAU( * ), WORK( * )
        function opmtr!(side::AbstractChar, uplo::AbstractChar, trans::AbstractChar, m::Integer, n::Integer,
            AP::PtrOrVec{$elty}, τ::PtrOrVec{$elty}, C::PtrOrMat{$elty}, ldc::Integer, work::PtrOrVec{$elty})
            chkuplo(uplo)
            info = Ref{BlasInt}()
            @blascall $opmtr(
                side::Ref{UInt8}, uplo::Ref{UInt8}, trans::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, AP::Ptr{$elty},
                τ::Ptr{$elty}, C::Ptr{$elty}, ldc::Ref{BlasInt}, work::Ptr{$elty}, info::Ref{BlasInt}, 1::Clong, 1::Clong,
                1::Clong
            )
            chklapackerror(info[])
            return C
        end
    end
end

macro pmalso(scaled, fun=nothing)
    if isnothing(fun)
        fun = scaled
        scaled = :disallow
    else
        scaled = scaled.value
    end
    @assert(scaled ∈ (:disallow, :unscale, :diagscale, :ignore))
    # fun is a function definition that contains uplo and a PM-typed parameter. We rewrite this into two function definitions:
    # - one where we replace all PM{T} by PM and add where {PM<:AbstractVector{T}}
    # - one where we remove the uplo parameter, and introduce the uplo variable
    #   if scaled === :disallow: replace all PM{T} by PM and add where {PM<:SPMatrixUnscaled{T}}
    #   if scaled === :unscale: replace all PM{T} by PM and add where {PM<:SPMatrix{T}}. Method should do packed_unscale!
    #                           somewhere
    #   if scaled === :diagscale: replace all PM{T} by PM, add where {PM<:SPMatrix{T}}, and set the variable
    #                             scalefac = sqrt(T(2)) if appropriate. Method should do
    #                             PM <: SPMatrixScaled && rmul_diags!(, scalefac) somewhere
    #   if scaled === :ignore, just allow all packed matrices without anything to add
    # In all functions, we also add the variables ...v for the vec() of the packed matrices (or the objects themselves).
    @assert(fun.head === :function && fun.args[2].head === :block)
    fun2 = deepcopy(fun)
    fncall, fncall2 = fun.args[1], fun2.args[1]
    fncallparent, fncall2parent = fun, fun2
    while fncall.head === :where
        fncallparent = fncall
        fncall = fncall.args[1]
        fncall2parent = fncall2
        fncall2 = fncall2.args[1]
    end
    @assert(fncall.head === :call)
    commontype = nothing # this will hold the name of our generic type, most likely :T
    for i in length(fncall.args):-1:2
        arg = fncall.args[i]
        if arg.head === :(::)
            length(arg.args) < 2 && continue
            if arg.args[1] === :uplo
                deleteat!(fncall2.args, i) # remove the uplo parameter from the second function definition
                continue
            end
            name = arg.args[1]
            type = arg.args[2]
            if type === :PM
            elseif type isa Expr && type.head === :curly && type.args[1] === :PM
                if isnothing(commontype)
                    commontype = type.args[2] # capture the type argument
                else
                    @assert(type.args[2] === commontype)
                end
                arg.args[2] = :PM # replace PM{T} by PM
                fncall2.args[i].args[2] = :PM
            else
                continue
            end
            pushfirst!(fun.args[2].args, :($(Symbol(name, "v")) = $name))
            pushfirst!(fun2.args[2].args, :($(Symbol(name, "v")) = vec($name)))
        elseif arg === :uplo # doesn't happen as upload is always qualified with AbstractChar, but anyway...
            deleteat!(fncall2.args, i)
        end
    end
    # in the second function definition, introduce a new where capture
    supertype = scaled ∈ (:disallow, :ignore) ? :SPMatrixUnscaled : :SPMatrix
    fncallparent.args[1] = Expr(:where, fncall, isnothing(commontype) ? :(PM<:AbstractVector) :
                                                                        :(PM<:AbstractVector{$commontype}))
    fncall2parent.args[1] = Expr(:where, fncall2, isnothing(commontype) ? :(PM<:$supertype) :
                                                                          :(PM<:$supertype{$commontype}))
    # in the second function definition, make the variable uplo available
    if scaled === :diagscale
        pushfirst!(fun2.args[2].args, :(
            if PM <: SPMatrixScaled
                scalefac = sqrt($(isnothing(commontype) ? :2 : :(_realtype($commontype)(2))))
            end
        ))
    end
    pushfirst!(fun2.args[2].args, :(uplo = packed_ulchar(PM)))
    esc(quote
        $fun
        $fun2
    end)
end

spmv!(α::Real, AP::SPMatrixUnscaled, args...) = spmv!(packed_ulchar(AP), α, vec(AP), args...)
@doc replace(@doc(spmv!).meta[:results][1].text[1],
    "    spmv!(uplo, α, AP, x, β, y)" => "    spmv!(uplo, α, AP::AbstractVector, x, β, y)
    spmv!(α, AP::SPMatrixUnscaled, x, β, y)") spmv!

hpmv!(α::Number, AP::SPMatrixUnscaled, args...) = hpmv!(packed_ulchar(AP), α, vec(AP), args...)
@doc replace(@doc(spmv!).meta[:results][1].text[1],
    "    hpmv!(uplo, α, AP, x, β, y)" => "    hpmv!(uplo, α, AP::AbstractVector, x, β, y)
    hpmv!(α, AP::SPMatrixUnscaled, x, β, y)") hpmv!

function spr!(α::Real, x::AbstractArray{T}, AP::SPMatrix{T}) where {T<:BlasReal}
    AP = packed_unscale!(AP)
    spr!(packed_ulchar(AP), α, x, vec(AP))
    return AP
end
@doc (replace(@doc(spr!).meta[:results][1].text[1],
    "    spr!(uplo, α, x, AP)" => "    spr!(uplo, α, x, AP::AbstractVector)
    spr!(α, x, AP::SPMatrix)") * warnunscale) spr!

@pmalso :unscale function hpr!(uplo::AbstractChar, α::Real, x::AbstractVector{T}, AP::PM{T}) where {T<:BlasComplex}
    require_one_based_indexing(APv, x)
    N = length(x)
    2length(APv) ≥ N*(N + 1) ||
        throw(DimensionMismatch(lazy"Packed symmetric matrix A has size smaller than length(x) = $(N)."))
    chkstride1(APv)
    px, stx = vec_pointer_stride(x, ArgumentError("input vector with 0 stride is not allowed"))
    PM <: SPMatrixScaled && (AP = packed_unscale!(AP))
    GC.@preserve x hpr!(uplo, N, T(α), px, stx, APv)
    return AP
end

"""
    hpr!(uplo, α, x, AP::AbstractVector)
    hpr!(α, x, AP::SPMatrix)

Update matrix ``A`` as ``A + \\alpha x x'``, where ``A`` is a Hermitian matrix provided in packed format `AP` and `x` is a
vector.

With `uplo = 'U'`, the vector `AP` must contain the upper triangular part of the Hermitian matrix packed sequentially, column
by column, so that `AP[1]` contains `A[1, 1]`, `AP[2]` and `AP[3]` contain `A[1, 2]` and `A[2, 2]` respectively, and so on.

With `uplo = 'L'`, the vector `AP` must contain the lower triangular part of the symmetric matrix packed sequentially, column
by column, so that `AP[1]` contains `A[1, 1]`, `AP[2]` and `AP[3]` contain `A[2, 1]` and `A[3, 1]` respectively, and so on.

The scalar input `α` must be real.

The array inputs `x` and `AP` must all be of `ComplexF32` or `ComplexF64` type. Return the updated `AP`.
$warnunscale
"""
hpr!

"""
    gemmt!(uplo, transA, transB, alpha, A, B, beta, C)

`gemmt!` computes a matrix-matrix product with general matrices but updates only the upper or lower triangular part of the
result matrix.

!!! info
    This function is a recent BLAS extension; for OpenBLAS, it requires at least version 0.3.22 (which is not yet shipped with
    Julia). If the currently available BLAS does not offer `gemmt`, the function falls back to `gemm`.
""" gemmt!

@pmalso function trttp!(uplo::AbstractChar, A::AbstractMatrix{T}, AP::PM{T}) where {T<:BlasFloat}
    chkstride1(APv)
    require_one_based_indexing(APv, A)
    n = checksquare(A)
    chkpacked(n, AP)
    trttp!(uplo, n, A, max(1, stride(A, 2)), APv)
    return AP
end

@pmalso function trttp!(uplo::AbstractChar, A::AbstractMatrix{T}, AP::PM{T}) where {T}
    chkuplo(uplo)
    n = checksquare(A)
    chkpacked(n, AP)
    if uplo == 'L' || uplo == 'l'
        k = 0
        @inbounds for j in 1:n, i in j:n
            k += 1
            APv[k] = A[i, j]
        end
    else
        k = 0
        @inbounds for j in 1:n, i in 1:j
            k += 1
            APv[k] = A[i, j]
        end
    end
    return AP
end

"""
    trttp!(uplo, A, AP::AbstractVector)
    trttp!(A, AP::SPMatrixUnscaled)

`trttp!` copies a triangular matrix from the standard full format (TR) to the standard packed format (TP).

!!! info
    This function is also implemented in plain Julia and therefore works with arbitrary element types.
"""
trttp!

@pmalso function tpttr!(uplo::AbstractChar, AP::PM{T}, A::AbstractMatrix{T}) where {T<:BlasFloat}
    chkstride1(APv)
    require_one_based_indexing(APv, A)
    n = checksquare(A)
    chkpacked(n, AP)
    return tpttr!(uplo, n, APv, A, max(1, stride(A, 2)))
end

@pmalso function tpttr!(uplo::AbstractChar, AP::PM{T}, A::AbstractMatrix{T}) where {T}
    chkuplo(uplo)
    n = checksquare(A)
    chkpacked(n, AP)
    if uplo == 'L' || uplo == 'l'
        k = 0
        @inbounds for j in 1:n, i in j:n
            k += 1
            A[i, j] = APv[k]
        end
    else
        k = 0
        @inbounds for j in 1:n, i in 1:j
            k += 1
            A[i, j] = APv[k]
        end
    end
    return A
end

"""
    tpttr!(uplo, AP::AbstractVector, A)
    tpttr!(AP::SPMatrixUnscaled, A)

`tpttr!` copies a triangular matrix from the standard packed format (TP) to the standard full format (TR).

!!! info
    This function is also implemented in plain Julia and therefore works with arbitrary element types.
"""
tpttr!

@pmalso :unscale function pptrf!(uplo::AbstractChar, AP::PM{<:BlasFloat})
    require_one_based_indexing(APv)
    chkstride1(APv)
    PM <: SPMatrixScaled && (AP = packed_unscale!(AP))
    _, info = pptrf!(uplo, packedside(AP), APv)
    return AP, info
end

"""
    pptrf!(uplo, AP::AbstractVector) -> (AP, info)
    pptrf!(AP::SPMatrix) -> (AP, info)

`pptrf!` computes the Cholesky factorization of a real symmetric or complex Hermitian positive definite matrix ``A`` stored in
packed format `AP`.

The factorization has the form
- ``A = U' U``,  if `uplo == 'U'`, or
- ``A = L L'``,  if `uplo == 'L'`,
where ``U`` is an upper triangular matrix and ``L`` is lower triangular.

If `info > 0`, the leading minor of order `info` is not positive definite, and the factorization could not be completed.
$warnunscale
"""
pptrf!

@pmalso function pptrs!(uplo::AbstractChar, AP::PM{T}, B::AbstractVecOrMat{T}) where {T<:BlasFloat}
    require_one_based_indexing(APv, B)
    chkstride1(APv, B)
    n = packedside(AP)
    nrhs = size(B, 2)
    size(B, 1) == n || throw(DimensionMismatch("first dimension of B, $(size(B,1)), and size of A, ($n,$n), must match!"))
    iszero(nrhs) && return B
    return pptrs!(uplo, n, nrhs, APv, B, max(1, stride(B, 2)))
end

"""
    pptrs!(uplo, AP::AbstractVector, B)
    pptrs!(AP::SPMatrixUnscaled, B)

`pptrs!` solves a system of linear equations ``A X = B`` with a symmetric or Hermitian positive definite matrix ``A`` in packed
storage `AP` using the Cholesky factorization ``A = U' U`` or ``A = L L'`` computed by [`pptrf!`](@ref).
"""
pptrs!

@pmalso function pptri!(uplo::AbstractChar, AP::PM{<:BlasFloat})
    require_one_based_indexing(APv)
    chkstride1(APv)
    pptri!(uplo, packedside(AP), APv)
    return AP
end

"""
    pptri!(uplo, AP::AbstractVector)
    pptri!(AP::SPMatrixUnscaled)

`pptri!` computes the inverse of a real symmetric or complex Hermitian positive definite matrix `A` using the Cholesky
factorization ``A = U' U`` or ``A = L L'`` computed by [`pptrf!`](@ref).
"""
pptri!

@doubledoc psv begin
    @pmalso :unscale function $f(uplo::AbstractChar, AP::PM{T}, ipiv::Union{<:AbstractVector{BlasInt},Missing},
        B::AbstractVecOrMat{T}) where {T<:$T}
        require_one_based_indexing(APv, B)
        chkstride1(APv, B)
        n = packedside(AP)
        size(B, 1) == n || throw(DimensionMismatch("B has first dimension $(size(B, 1)), but needs $n"))
        if ismissing(ipiv)
            ipiv = Vector{BlasInt}(undef, n)
        else
            length(ipiv) == n || throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $n"))
            require_one_based_indexing(ipiv)
            chkstride1(ipiv)
        end
        PM <: SPMatrixScaled && (AP = packed_unscale!(AP))
        $f(uplo, n, size(B, 2), APv, ipiv, B, max(1, stride(B, 2)))
        return B, AP, ipiv
    end
    @pmalso :unscale function $f(uplo::AbstractChar, AP::PM{T}, B::AbstractVecOrMat{T}) where {T<:$T}
        return $f(uplo, AP, missing, B)
    end

"""
    $($fn)(uplo, AP::AbstractVector, ipiv=missing, B) -> (B, AP, ipiv)
    $($fn)(AP::SPMatrix, ipiv=missing, B) -> (B, AP, ipiv)

`$($fn)` computes the solution to a $($field) system of linear equations ``A X = B``, where ``A`` is an ``N``-by-``N`` $($typ)
matrix stored in packed format `AP` and ``X`` and `B` are ``N``-by-``N_{\\mathrm{rhs}}`` matrices.

The diagonal pivoting method is used to factor ``A`` as
- ``A = U D U$($op)``,  if `UPLO = 'U'`, or
- ``A = L D L$($op)``,  if `UPLO = 'L'`,
where `U` (or `L`) is a product of permutation and unit upper (lower) triangular matrices, `D` is $($typ) and block diagonal
with 1-by-1 and 2-by-2 diagonal blocks. The factored form of ``A`` is then used to solve the system of equations ``A X = B``.
$($warnunscale)
"""
    $f
end

@doubledoc ptrf begin
    @pmalso :unscale function $f(uplo::AbstractChar, AP::PM{<:$T}, ipiv::Union{<:AbstractVector{BlasInt},Missing}=missing)
        require_one_based_indexing(APv)
        chkstride1(APv)
        n = packedside(AP)
        if ismissing(ipiv)
            ipiv = Vector{BlasInt}(undef, n)
        else
            length(ipiv) == n || throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $n"))
            require_one_based_indexing(ipiv)
            chkstride1(ipiv)
        end
        PM <: SPMatrixScaled && (AP = packed_unscale!(AP))
        _, _, info = $f(uplo, n, APv, ipiv)
        return AP, ipiv, info
    end

"""
    $($fn)(uplo, AP::AbstractVector, ipiv=missing) -> (AP, ipiv, info)
    $($fn)(AP::SPMatrix, ipiv=missing) -> (AP, ipiv, info)

`$($fn)` computes the factorization of a $($field) $($typ) matrix ``A`` stored in packed format `AP` using the Bunch-Kaufman
diagonal pivoting method:

``A = U D U$($op)``  or  ``A = L D L$($op)``

where ``U`` (or ``L``) is a product of permutation and unit upper (lower) triangular matrices, and ``D`` is $($typ) and block
diagonal with 1-by-1 and 2-by-2 diagonal blocks.
$($warnunscale)
"""
    $f
end

@doubledoc ptrs begin
    @pmalso function $f(uplo::AbstractChar, AP::PM{T}, ipiv::AbstractVector{BlasInt}, B::AbstractVecOrMat{T}) where {T<:$T}
        require_one_based_indexing(APv, B, ipiv)
        chkstride1(APv, B, ipiv)
        n = length(ipiv)
        chkpacked(n, AP)
        size(B, 1) == n || throw(DimensionMismatch("B has first dimension $(size(B, 1)), but needs $n"))
        return $f(uplo, n, size(B, 2), APv, ipiv, B, max(1, stride(B, 2)))
    end

"""
    $($fn)(uplo, AP::AbstractVector, ipiv, B)
    $($fn)(AP::SPMatrixUnscaled, ipiv, B)

`$($fn)` solves a system of linear equations ``A X = B`` with a $($field) $($typ) matrix ``A`` stored in packed format using
the factorization ``A = U D U$($op)`` or ``A = L D L$($op)`` computed by [`$($prefix)ptrf!`](@ref).
"""
    $f
end

@doubledoc ptri begin
    @pmalso function $f(uplo::AbstractChar, AP::PM{T}, ipiv::AbstractVector{BlasInt},
        work::Union{<:AbstractVector{T},Missing}=missing) where {T<:$T}
        require_one_based_indexing(APv, ipiv)
        chkstride1(APv, ipiv)
        n = length(ipiv)
        chkpacked(n, AP)
        if ismissing(work)
            work = Vector{T}(undef, n)
        else
            length(work) ≥ n || throw(DimensionMismatch("work has length $(length(work)), but needs at least $n"))
            require_one_based_indexing(work)
            chkstride1(work)
        end
        $f(uplo, n, APv, ipiv, work)
        return AP
    end

"""
    $($fn)(uplo, AP::AbstractVector, ipiv, work=missing)
    $($fn)(AP::SPMatrixUnscaled, ipiv, work=missing)

`$($fn)` computes the inverse of a $($field) $($typ) indefinite matrix ``A`` in packed storage `AP` using the factorization
``A = U D U$($op)`` or ``A = L D L$($op)`` computed by [`$($prefix)ptrf!`](@ref).
"""
    $f
end

spev!(jobz::AbstractChar, args...) = spev!(Val(Symbol(jobz)), args...)

@pmalso :diagscale function spev!(::Val{:N}, uplo::AbstractChar, AP::PM{T}, W::Union{<:AbstractVector{T},Missing}=missing,
    work::Union{<:AbstractVector{T},Missing}=missing) where {T<:BlasReal}
    require_one_based_indexing(APv)
    chkstride1(APv)
    if ismissing(W)
        n = packedside(AP)
        W = Vector{T}(undef, n)
    else
        n = length(W)
        chkpacked(n, AP)
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(work)
        work = Vector{T}(undef, 3n)
    else
        length(work) < 3n && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    PM <: SPMatrixScaled && rmul_diags!(AP, scalefac)
    evs, _ = spev!('N', uplo, n, APv, W, Ptr{T}(C_NULL), 1, work)
    return PM <: SPMatrixScaled ? rmul!(evs, inv(scalefac)) : evs
end

@pmalso :diagscale function spev!(::Val{:V}, uplo::AbstractChar, AP::PM{T}, W::Union{<:AbstractVector{T},Missing}=missing,
    Z::Union{<:AbstractMatrix{T},Missing}=missing, work::Union{<:AbstractVector{T},Missing}=missing) where {T<:BlasReal}
    require_one_based_indexing(APv)
    chkstride1(APv)
    if ismissing(W)
        n = packedside(AP)
        W = Vector{T}(undef, n)
    else
        n = length(W)
        chkpacked(n, AP)
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(Z)
        Z = Matrix{T}(undef, n, n)
    else
        checksquare(Z) == n || throw(DimensionMismatch("The provided matrix was too small"))
        require_one_based_indexing(Z)
        chkstride1(Z)
    end
    if ismissing(work)
        work = Vector{T}(undef, 3n)
    else
        length(work) < 3n && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    PM <: SPMatrixScaled && rmul_diags!(AP, scalefac)
    evs, evecs = spev!('V', uplo, n, APv, W, Z, max(1, stride(Z, 2)), work)
    return (PM <: SPMatrixScaled ? rmul!(evs, inv(scalefac)) : evs), evecs
end

@pmalso :diagscale function spev!(::Val{:N}, uplo::AbstractChar, AP::PM{T}, W::Union{<:AbstractVector{R},Missing}=missing,
    work::Union{<:AbstractVector{T},Missing}=missing,
    rwork::Union{<:AbstractVector{R},Missing}=missing) where {R<:BlasReal,T<:Complex{R}}
    require_one_based_indexing(APv)
    chkstride1(APv)
    if ismissing(W)
        n = packedside(AP)
        W = Vector{R}(undef, n)
    else
        n = length(W)
        chkpacked(n, AP)
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(work)
        work = Vector{T}(undef, max(1, 2n -1))
    else
        length(work) < max(1, 2n -1) && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(rwork)
        rwork = Vector{R}(undef, max(1, 3n -2))
    else
        length(rwork) < max(1, 3n -2) && throw(ArgumentError("The provided rwork space was too small"))
        require_one_based_indexing(rwork)
        chkstride1(rwork)
    end
    PM <: SPMatrixScaled && rmul_diags!(AP, scalefac)
    evs, _ = spev!('N', uplo, n, APv, W, Ptr{T}(C_NULL), 1, work, rwork)
    return PM <: SPMatrixScaled ? rmul!(evs, inv(scalefac)) : evs
end

@pmalso :diagscale function spev!(::Val{:V}, uplo::AbstractChar, AP::PM{T}, W::Union{<:AbstractVector{R},Missing}=missing,
    Z::Union{<:AbstractMatrix{T},Missing}=missing, work::Union{<:AbstractVector{T},Missing}=missing,
    rwork::Union{<:AbstractVector{R},Missing}=missing) where {R<:BlasReal,T<:Complex{R}}
    require_one_based_indexing(APv)
    chkstride1(APv)
    if ismissing(W)
        n = packedside(AP)
        W = Vector{R}(undef, n)
    else
        n = length(W)
        chkpacked(n, AP)
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(Z)
        Z = Matrix{T}(undef, n, n)
    else
        checksquare(Z) == n || throw(DimensionMismatch("The provided matrix was too small"))
        require_one_based_indexing(Z)
        chkstride1(Z)
    end
    if ismissing(work)
        work = Vector{T}(undef, max(1, 2n -1))
    else
        length(work) < max(1, 2n -1) && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(rwork)
        rwork = Vector{R}(undef, max(1, 3n -2))
    else
        length(rwork) < max(1, 3n -2) && throw(ArgumentError("The provided rwork space was too small"))
        require_one_based_indexing(rwork)
        chkstride1(rwork)
    end
    PM <: SPMatrixScaled && rmul_diags!(AP, scalefac)
    evs, evecs = spev!('V', uplo, n, APv, W, Z, max(1, stride(Z, 2)), work, rwork)
    return (PM <: SPMatrixScaled ? rmul!(evs, inv(scalefac)) : evs), evecs
end

"""
    spev!('N', uplo, AP::AbstractVector, W=missing, work=missing[, rwork=missing]) -> W
    spev!('N', AP::SPMatrix, W=missing, work=missing[, rwork=missing]) -> W
    spev!('V', uplo, AP::AbstractVector, W=missing, Z=missing, work=missing
        [, rwork=missing]) -> (W, Z)
    spev!('V', AP::SPMatrix, W=missing, Z=missing, work=missing[, rwork=missing])
        -> (W, Z)

Finds the eigenvalues (first parameter `'N'`) or eigenvalues and eigenvectors (first parameter `'V'`) of a Hermitian matrix
``A`` in packed storage `AP`. If `uplo = 'U'`, `AP` is the upper triangle of ``A``. If `uplo = 'L'`, `AP` is the lower triangle
of ``A``.

The output vector `W` and matrix `Z`, as well as the temporaries `work` and `rwork` (in the complex case) may be specified to
save allocations.
"""
spev!

spevx!(jobz::AbstractChar, args...) = spevx!(Val(Symbol(jobz)), args...)
spevx!(jobz::Val, uplo::AbstractChar, AP::AbstractVector) =
    spevx!(jobz, 'A', uplo, AP, nothing, nothing, nothing, nothing, _realtype(eltype(AP))(-1))
spevx!(jobz::Val, AP::SPMatrix) =
    spevx!(jobz, 'A', AP, nothing, nothing, nothing, nothing, _realtype(eltype(AP))(-1))

@pmalso :diagscale function spevx!(::Val{:N}, range::AbstractChar, uplo::AbstractChar, AP::PM{T}, vl::Union{Nothing,T},
    vu::Union{Nothing,T}, il::Union{Nothing,<:Integer}, iu::Union{Nothing,<:Integer}, abstol::T,
    W::Union{<:AbstractVector{T},Missing}=missing, work::Union{<:AbstractVector{T},Missing}=missing,
    iwork::Union{<:AbstractVector{BlasInt},Missing}=missing) where {T<:BlasReal}
    @nospecialize vl vu il iu
    require_one_based_indexing(APv)
    chkstride1(APv)
    n = packedside(AP)
    if ismissing(W)
        W = Vector{T}(undef, range == 'I' ? iu - il + 1 : n)
    else
        if range == 'A'
            length(W) ≥ n || throw(DimensionMismatch("W has length $(length(W)), but needs at least $n"))
        elseif range == 'I'
            length(W) ≥ iu - il +1 || throw(DimensionMismatch("W has length $(length(W)), but needs at least $(iu - il +1)"))
        end
        # we cannot check 'V'
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(work)
        work = Vector{T}(undef, 8n)
    else
        length(work) < 8n && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(iwork)
        iwork = Vector{BlasInt}(undef, 5n)
    else
        length(iwork) < 5n && throw(ArgumentError("The provided iwork space was too small"))
        require_one_based_indexing(iwork)
        chkstride1(iwork)
    end
    if PM <: SPMatrixScaled
        rmul_diags!(AP, scalefac)
        !isnothing(vl) && (vl *= scalefac)
        !isnothing(vu) && (vu *= scalefac)
    end
    m, _, _, _, _ = spevx!('N', range, uplo, n, APv, vl, vu, il, iu, abstol, W, Ptr{T}(C_NULL), 1, work, iwork,
        Ptr{BlasInt}(C_NULL))
    return PM <: SPMatrixScaled ? rmul!(@view(W[1:m]), inv(scalefac)) : @view(W[1:m])
end

@pmalso :diagscale function spevx!(::Val{:V}, range::AbstractChar, uplo::AbstractChar, AP::PM{T}, vl::Union{Nothing,T},
    vu::Union{Nothing,T}, il::Union{Nothing,<:Integer}, iu::Union{Nothing,<:Integer}, abstol::T,
    W::Union{<:AbstractVector{T},Missing}=missing, Z::Union{<:AbstractMatrix{T},Missing}=missing,
    work::Union{<:AbstractVector{T},Missing}=missing,
    iwork::Union{<:AbstractVector{BlasInt},Missing}=missing,
    ifail::Union{<:AbstractVector{BlasInt},Missing}=missing) where {T<:BlasReal}
    @nospecialize vl vu il iu
    require_one_based_indexing(APv)
    chkstride1(APv)
    n = packedside(AP)
    if ismissing(W)
        W = Vector{T}(undef, range == 'I' ? iu - il + 1 : n)
    else
        if range == 'A'
            length(W) ≥ n || throw(DimensionMismatch("W has length $(length(W)), but needs at least $n"))
        elseif range == 'I'
            length(W) ≥ iu - il +1 || throw(DimensionMismatch("W has length $(length(W)), but needs at least $(iu - il +1)"))
        end
        # we cannot check 'V'
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(Z)
        Z = Matrix{T}(undef, n, range == 'I' ? iu - il + 1 : n)
    else
        size(Z, 1) == n || throw(DimensionMismatch("Z has first dimension $(size(Z, 1)), but needs $n"))
        if range == 'A'
            size(Z, 2) ≥ n || throw(DimensionMismatch("Z has second dimension $(size(Z, 2)), but needs $n"))
        elseif range == 'I'
            size(Z, 2) ≥ iu - il +1 ||
                throw(DimensionMismatch("Z has second dimension $(size(Z, 2)), but needs at least $(iu - il +1)"))
        end
        require_one_based_indexing(Z)
        chkstride1(Z)
    end
    if ismissing(work)
        work = Vector{T}(undef, 8n)
    else
        length(work) < 8n && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(iwork)
        iwork = Vector{BlasInt}(undef, 5n)
    else
        length(iwork) < 5n && throw(ArgumentError("The provided iwork space was too small"))
        require_one_based_indexing(iwork)
        chkstride1(iwork)
    end
    if ismissing(ifail)
        ifail = Vector{BlasInt}(undef, n)
    else
        length(ifail) < n && throw(DimensionMismatch("The provided ifail space was too small"))
        require_one_based_indexing(ifail)
        chkstride1(ifail)
    end
    if PM <: SPMatrixScaled
        rmul_diags!(AP, scalefac)
        !isnothing(vl) && (vl *= scalefac)
        !isnothing(vu) && (vu *= scalefac)
    end
    m, _, _, info, _ = spevx!('V', range, uplo, n, APv, vl, vu, il, iu, abstol, W, Z, max(1, stride(Z, 2)), work, iwork, ifail)
    return (PM <: SPMatrixScaled ? rmul!(@view(W[1:m]), inv(scalefac)) : @view(W[1:m])),
        @view(Z[:, 1:m]), info, @view(ifail[1:m])
end

@pmalso :diagscale function spevx!(::Val{:N}, range::AbstractChar, uplo::AbstractChar, AP::PM{T}, vl::Union{Nothing,R},
    vu::Union{Nothing,R}, il::Union{Nothing,<:Integer}, iu::Union{Nothing,<:Integer}, abstol::R,
    W::Union{<:AbstractVector{R},Missing}=missing,
    work::Union{<:AbstractVector{T},Missing}=missing, rwork::Union{<:AbstractVector{R},Missing}=missing,
    iwork::Union{<:AbstractVector{BlasInt},Missing}=missing) where {R<:BlasReal,T<:Complex{R}}
    @nospecialize vl vu il iu
    require_one_based_indexing(APv)
    chkstride1(APv)
    n = packedside(AP)
    if ismissing(W)
        W = Vector{R}(undef, range == 'I' ? iu - il + 1 : n)
    else
        if range == 'A'
            length(W) ≥ n || throw(DimensionMismatch("W has length $(length(W)), but needs at least $n"))
        elseif range == 'I'
            length(W) ≥ iu - il +1 || throw(DimensionMismatch("W has length $(length(W)), but needs at least $(iu - il +1)"))
        end
        # we cannot check 'V'
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(work)
        work = Vector{T}(undef, 2n)
    else
        length(work) < 2n && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(rwork)
        rwork = Vector{R}(undef, 7n)
    else
        length(rwork) < 7n && throw(ArgumentError("The provided rwork space was too small"))
        require_one_based_indexing(rwork)
        chkstride1(rwork)
    end
    if ismissing(iwork)
        iwork = Vector{BlasInt}(undef, 5n)
    else
        length(iwork) < 5n && throw(ArgumentError("The provided iwork space was too small"))
        require_one_based_indexing(iwork)
        chkstride1(iwork)
    end
    if PM <: SPMatrixScaled
        rmul_diags!(AP, scalefac)
        !isnothing(vl) && (vl *= scalefac)
        !isnothing(vu) && (vu *= scalefac)
    end
    m, _, _, _, _, = spevx!('N', range, uplo, n, APv, vl, vu, il, iu, abstol, W, Ptr{T}(C_NULL), 1, work, rwork, iwork,
        Ptr{BlasInt}(C_NULL))
    return PM <: SPMatrixScaled ? rmul!(@view(W[1:m]), inv(scalefac)) : @view(W[1:m])
end

@pmalso :diagscale function spevx!(::Val{:V}, range::AbstractChar, uplo::AbstractChar, AP::PM{T}, vl::Union{Nothing,R},
    vu::Union{Nothing,R}, il::Union{Nothing,<:Integer}, iu::Union{Nothing,<:Integer}, abstol::R,
    W::Union{<:AbstractVector{R},Missing}=missing, Z::Union{<:AbstractMatrix{T},Missing}=missing,
    work::Union{<:AbstractVector{T},Missing}=missing, rwork::Union{<:AbstractVector{R},Missing}=missing,
    iwork::Union{<:AbstractVector{BlasInt},Missing}=missing,
    ifail::Union{<:AbstractVector{BlasInt},Missing}=missing) where {R<:BlasReal,T<:Complex{R}}
    @nospecialize vl vu il iu
    require_one_based_indexing(APv)
    chkstride1(APv)
    n = packedside(AP)
    if ismissing(W)
        W = Vector{R}(undef, range == 'I' ? iu - il + 1 : n)
    else
        if range == 'A'
            length(W) ≥ n || throw(DimensionMismatch("W has length $(length(W)), but needs at least $n"))
        elseif range == 'I'
            length(W) ≥ iu - il +1 || throw(DimensionMismatch("W has length $(length(W)), but needs at least $(iu - il +1)"))
        end
        # we cannot check 'V'
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(Z)
        Z = Matrix{T}(undef, n, range == 'I' ? iu - il + 1 : n)
    else
        size(Z, 1) == n || throw(DimensionMismatch("Z has first dimension $(size(Z, 1)), but needs $n"))
        if range == 'A'
            size(Z, 2) ≥ n || throw(DimensionMismatch("Z has second dimension $(size(Z, 2)), but needs $n"))
        elseif range == 'I'
            size(Z, 2) ≥ iu - il +1 ||
                throw(DimensionMismatch("Z has second dimension $(size(Z, 2)), but needs at least $(iu - il +1)"))
        end
        require_one_based_indexing(Z)
        chkstride1(Z)
    end
    if ismissing(work)
        work = Vector{T}(undef, 2n)
    else
        length(work) < 2n && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(rwork)
        rwork = Vector{R}(undef, 7n)
    else
        length(rwork) < 7n && throw(ArgumentError("The provided rwork space was too small"))
        require_one_based_indexing(rwork)
        chkstride1(rwork)
    end
    if ismissing(iwork)
        iwork = Vector{BlasInt}(undef, 5n)
    else
        length(iwork) < 5n && throw(ArgumentError("The provided iwork space was too small"))
        require_one_based_indexing(iwork)
        chkstride1(iwork)
    end
    if ismissing(ifail)
        ifail = Vector{BlasInt}(undef, n)
    else
        length(ifail) < n && throw(DimensionMismatch("The provided ifail space was too small"))
        require_one_based_indexing(ifail)
        chkstride1(ifail)
    end
    if PM <: SPMatrixScaled
        rmul_diags!(AP, scalefac)
        !isnothing(vl) && (vl *= scalefac)
        !isnothing(vu) && (vu *= scalefac)
    end
    m, _, _, info, _ = spevx!('V', range, uplo, n, APv, vl, vu, il, iu, abstol, W, Z, max(1, stride(Z, 2)), work, rwork, iwork,
        ifail)
    return (PM <: SPMatrixScaled ? rmul!(@view(W[1:m]), inv(scalefac)) : @view(W[1:m])),
        @view(Z[:, 1:m]), info, @view(ifail[1:m])
end

"""
    spevx!('N', range, uplo, AP::AbstractVector, vl, vu, il, iu, abstol, W=missing,
        work=missing[, rwork=missing], iwork=missing) -> view(W)
    spevx!('N', range, AP::SPMatrix, vl, vu, il, iu, abstol, W=missing, work=missing
        [, rwork=missing], iwork=missing) -> view(W)
    spevx!('V', range, uplo, AP::AbstractVector, vl, vu, il, iu, abstol, W=missing,
        Z=missing, work=missing[, rwork=missing], iwork=missing, ifail=missing)
        -> (view(W), view(Z), info, view(ifail))
    spevx!('V', range, AP::SPMatrix, vl, vu, il, iu, abstol, W=missing, Z=missing,
        work=missing[, rwork=missing], iwork=missing, ifail=missing)
        -> (view(W), view(Z), info, view(ifail))

Finds the eigenvalues (first parameter `'N'`) or eigenvalues and eigenvectors (first parameter `'V'`) of a Hermitian matrix
``A`` in packed storage `AP`. If `uplo = 'U'`, `AP` is the upper triangle of ``A``. If `uplo = 'L'`, `AP` is the lower triangle
of ``A``. If `range = 'A'`, all the eigenvalues are found. If `range = 'V'`, the eigenvalues in the half-open interval
`(vl, vu]` are found. If `range = 'I'`, the eigenvalues with indices between `il` and `iu` are found. `abstol` can be set as a
tolerance for convergence.

The eigenvalues are returned in `W` and the eigenvectors in `Z`. `info` is zero upon success or the number of eigenvalues that
failed to converge; their indices are stored in `ifail`. The resulting views to the original data are sized according to the
number of eigenvalues that fulfilled the bounds given by the range.

The output vector `W`, `ifail` and matrix `Z`, as well as the temporaries `work`, `rwork` (in the complex case), and `iwork`
may be specified to save allocations.
"""
spevx!

spevd!(jobz::AbstractChar, args...) = spevd!(Val(Symbol(jobz)), args...)

@pmalso :diagscale function spevd!(::Val{:N}, uplo::AbstractChar, AP::PM{T}, W::Union{<:AbstractVector{T},Missing}=missing,
    work::Union{<:AbstractVector{T},Missing}=missing,
    iwork::Union{<:AbstractVector{BlasInt},Missing}=missing) where {T<:BlasReal}
    require_one_based_indexing(APv)
    chkstride1(APv)
    if ismissing(W)
        n = packedside(AP)
        W = Vector{T}(undef, n)
    else
        n = length(W)
        chkpacked(n, AP)
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(work)
        work = Vector{T}(undef, n ≤ 1 ? 1 : 2n)
    else
        length(work) < (n ≤ 1 ? 1 : 2n) && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(iwork)
        iwork = Vector{BlasInt}(undef, 1)
    else
        length(iwork) < 1 && throw(ArgumentError("The provided iwork space was too small"))
        require_one_based_indexing(iwork)
        chkstride1(iwork)
    end
    PM <: SPMatrixScaled && rmul_diags!(AP, scalefac)
    evs, _ = spevd!('N', uplo, n, APv, W, Ptr{T}(C_NULL), 1, work, length(work), iwork, length(iwork))
    return PM <: SPMatrixScaled ? rmul!(evs, inv(scalefac)) : evs
end

@pmalso :diagscale function spevd!(::Val{:V}, uplo::AbstractChar, AP::PM{T}, W::Union{<:AbstractVector{T},Missing}=missing,
    Z::Union{<:AbstractMatrix{T},Missing}=missing, work::Union{<:AbstractVector{T},Missing}=missing,
    iwork::Union{<:AbstractVector{BlasInt},Missing}=missing) where {T<:BlasReal}
    require_one_based_indexing(APv)
    chkstride1(APv)
    if ismissing(W)
        n = packedside(AP)
        W = Vector{T}(undef, n)
    else
        n = length(W)
        chkpacked(n, AP)
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(Z)
        Z = Matrix{T}(undef, n, n)
    else
        checksquare(Z) == n || throw(DimensionMismatch("The provided matrix was too small"))
        require_one_based_indexing(Z)
        chkstride1(Z)
    end
    if ismissing(work)
        work = Vector{T}(undef, n ≤ 1 ? 1 : n^2 + 6n +1)
    else
        length(work) < (n ≤ 1 ? 1 : n^2 + 6n +1) && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(iwork)
        iwork = Vector{BlasInt}(undef, 3 + 5n)
    else
        length(iwork) < 3 + 5n && throw(ArgumentError("The provided iwork space was too small"))
        require_one_based_indexing(iwork)
        chkstride1(iwork)
    end
    PM <: SPMatrixScaled && rmul_diags!(AP, scalefac)
    evs, evecs = spevd!('V', uplo, n, APv, W, Z, max(1, stride(Z, 2)), work, length(work), iwork, length(iwork))
    return (PM <: SPMatrixScaled ? rmul!(evs, inv(scalefac)) : evs), evecs
end

@pmalso :diagscale function spevd!(::Val{:N}, uplo::AbstractChar, AP::PM{T}, W::Union{<:AbstractVector{R},Missing}=missing,
    work::Union{<:AbstractVector{T},Missing}=missing, rwork::Union{<:AbstractVector{R},Missing}=missing,
    iwork::Union{<:AbstractVector{BlasInt},Missing}=missing) where {R<:BlasReal,T<:Complex{R}}
    require_one_based_indexing(APv)
    chkstride1(APv)
    if ismissing(W)
        n = packedside(AP)
        W = Vector{R}(undef, n)
    else
        n = length(W)
        chkpacked(n, AP)
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(work)
        work = Vector{T}(undef, n ≤ 1 ? 1 : n)
    else
        length(work) < (n ≤ 1 ? 1 : n) && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(rwork)
        rwork = Vector{R}(undef, n ≤ 1 ? 1 : n)
    else
        length(rwork) < (n ≤ 1 ? 1 : n) && throw(ArgumentError("The provided rwork space was too small"))
        require_one_based_indexing(rwork)
        chkstride1(rwork)
    end
    if ismissing(iwork)
        iwork = Vector{BlasInt}(undef, 1)
    else
        length(iwork) < 1 && throw(ArgumentError("The provided iwork space was too small"))
        require_one_based_indexing(iwork)
        chkstride1(iwork)
    end
    PM <: SPMatrixScaled && rmul_diags!(AP, scalefac)
    evs, _ = spevd!('N', uplo, n, APv, W, Ptr{T}(C_NULL), 1, work, length(work), rwork, length(rwork), iwork, length(iwork))
    return PM <: SPMatrixScaled ? rmul!(evs, inv(scalefac)) : evs
end

@pmalso :diagscale function spevd!(::Val{:V}, uplo::AbstractChar, AP::PM{T}, W::Union{<:AbstractVector{R},Missing}=missing,
    Z::Union{<:AbstractMatrix{T},Missing}=missing, work::Union{<:AbstractVector{T},Missing}=missing,
    rwork::Union{<:AbstractVector{R},Missing}=missing,
    iwork::Union{<:AbstractVector{BlasInt},Missing}=missing) where {R<:BlasReal,T<:Complex{R}}
    require_one_based_indexing(APv)
    chkstride1(APv)
    if ismissing(W)
        n = packedside(AP)
        W = Vector{R}(undef, n)
    else
        n = length(W)
        chkpacked(n, AP)
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(Z)
        Z = Matrix{T}(undef, n, n)
    else
        checksquare(Z) == n || throw(DimensionMismatch("The provided matrix was too small"))
        require_one_based_indexing(Z)
        chkstride1(Z)
    end
    if ismissing(work)
        work = Vector{T}(undef, n ≤ 1 ? 1 : 2n)
    else
        length(work) < (n ≤ 1 ? 1 : 2n) && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(rwork)
        rwork = Vector{R}(undef, n ≤ 1 ? 1 : 2n^2 + 5n +1)
    else
        length(rwork) < (n ≤ 1 ? 1 : 2n^2 + 5n +1) && throw(ArgumentError("The provided rwork space was too small"))
        require_one_based_indexing(rwork)
        chkstride1(rwork)
    end
    if ismissing(iwork)
        iwork = Vector{BlasInt}(undef, 3 + 5n)
    else
        length(iwork) < 3 + 5n && throw(ArgumentError("The provided iwork space was too small"))
        require_one_based_indexing(iwork)
        chkstride1(iwork)
    end
    PM <: SPMatrixScaled && rmul_diags!(AP, scalefac)
    evs, evecs = spevd!('V', uplo, n, APv, W, Z, max(1, stride(Z, 2)), work, length(work), rwork, length(rwork), iwork,
        length(iwork))
    return (PM <: SPMatrixScaled ? rmul!(evs, inv(scalefac)) : evs), evecs
end

"""
    spevd!('N', uplo, AP::AbstractVector, W=missing, work=missing[, rwork=missing],
        iwork=missing) -> W
    spevd!('N', AP::SPMatrix, W=missing, work=missing[, rwork=missing],
        iwork=missing) -> W
    spevd!('V', uplo, AP::AbstractVector, W=missing, Z=missing, work=missing
        [, rwork=missing], iwork=missing) -> (W, Z)
    spevd!('V', AP::SPMatrix, W=missing, Z=missing, work=missing[, rwork=missing],
        iwork=missing) -> (W, Z)

Finds the eigenvalues (first parameter `'N'`) or eigenvalues and eigenvectors (first parameter `'V'`) of a Hermitian matrix
``A`` in packed storage `AP`. If `uplo = 'U'`, `AP` is the upper triangle of ``A``. If `uplo = 'L'`, `AP` is the lower triangle
of ``A``.

The output vector `W` and matrix `Z`, as well as the temporaries `work`, `rwork` (in the complex case), and `iwork` may be
specified to save allocations.

If eigenvectors are desired, it uses a divide and conquer algorithm.
"""
spevd!

spgv!(itype::Integer, jobz::AbstractChar, args...) = spgv!(itype, Val(Symbol(jobz)), args...)

@pmalso :unscale function spgv!(itype::Integer, ::Val{:N}, uplo::AbstractChar, AP::PM{T}, BP::PM{T},
    W::Union{<:AbstractVector{T},Missing}=missing, work::Union{<:AbstractVector{T},Missing}=missing) where {T<:BlasReal}
    require_one_based_indexing(APv, BPv)
    chkstride1(APv, BPv)
    length(APv) == length(BPv) || throw(DimensionMismatch("The dimensions of AP and BP must be the same"))
    if ismissing(W)
        n = packedside(AP)
        W = Vector{T}(undef, n)
    else
        n = length(W)
        chkpacked(n, AP)
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(work)
        work = Vector{T}(undef, 3n)
    else
        length(work) < 3n && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if PM <: SPMatrixScaled
        AP = packed_unscale!(AP)
        BP = packed_unscale!(BP)
    end
    evs, _ = spgv!(itype, 'N', uplo, n, APv, BPv, W, Ptr{T}(C_NULL), 1, work)
    return evs, BP
end

@pmalso :unscale function spgv!(itype::Integer, ::Val{:V}, uplo::AbstractChar, AP::PM{T}, BP::PM{T},
    W::Union{<:AbstractVector{T},Missing}=missing, Z::Union{<:AbstractMatrix{T},Missing}=missing,
    work::Union{<:AbstractVector{T},Missing}=missing) where {T<:BlasReal}
    require_one_based_indexing(APv, BPv)
    chkstride1(APv, BPv)
    length(APv) == length(BPv) || throw(DimensionMismatch("The dimensions of AP and BP must be the same"))
    if ismissing(W)
        n = packedside(AP)
        W = Vector{T}(undef, n)
    else
        n = length(W)
        chkpacked(n, AP)
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(Z)
        Z = Matrix{T}(undef, n, n)
    else
        checksquare(Z) == n || throw(DimensionMismatch("The provided matrix was too small"))
        require_one_based_indexing(Z)
        chkstride1(Z)
    end
    if ismissing(work)
        work = Vector{T}(undef, 3n)
    else
        length(work) < 3n && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if PM <: SPMatrixScaled
        AP = packed_unscale!(AP)
        BP = packed_unscale!(BP)
    end
    evs, evecs = spgv!(itype, 'V', uplo, n, APv, BPv, W, Z, max(1, stride(Z, 2)), work)
    return evs, evecs, BP
end

@pmalso :unscale function spgv!(itype::Integer, ::Val{:N}, uplo::AbstractChar, AP::PM{T}, BP::PM{T},
    W::Union{<:AbstractVector{R},Missing}=missing, work::Union{<:AbstractVector{T},Missing}=missing,
    rwork::Union{<:AbstractVector{R},Missing}=missing) where {R<:BlasReal,T<:Complex{R}}
    require_one_based_indexing(APv, BPv)
    chkstride1(APv, BPv)
    length(APv) == length(BPv) || throw(DimensionMismatch("The dimensions of AP and BP must be the same"))
    if ismissing(W)
        n = packedside(AP)
        W = Vector{R}(undef, n)
    else
        n = length(W)
        chkpacked(n, AP)
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(work)
        work = Vector{T}(undef, max(1, 2n -1))
    else
        length(work) < max(1, 2n -1) && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(rwork)
        rwork = Vector{R}(undef, max(1, 3n -2))
    else
        length(rwork) < max(1, 3n -2) && throw(ArgumentError("The provided rwork space was too small"))
        require_one_based_indexing(rwork)
        chkstride1(rwork)
    end
    if PM <: SPMatrixScaled
        AP = packed_unscale!(AP)
        BP = packed_unscale!(BP)
    end
    evs, _ = spgv!(itype, 'N', uplo, n, APv, BPv, W, Ptr{T}(C_NULL), 1, work, rwork)
    return evs, BP
end

@pmalso :unscale function spgv!(itype::Integer, ::Val{:V}, uplo::AbstractChar, AP::PM{T}, BP::PM{T},
    W::Union{<:AbstractVector{R},Missing}=missing, Z::Union{<:AbstractMatrix{T},Missing}=missing,
    work::Union{<:AbstractVector{T},Missing}=missing,
    rwork::Union{<:AbstractVector{R},Missing}=missing) where {R<:BlasReal,T<:Complex{R}}
    require_one_based_indexing(APv, BPv)
    chkstride1(APv, BPv)
    length(APv) == length(BPv) || throw(DimensionMismatch("The dimensions of AP and BP must be the same"))
    if ismissing(W)
        n = packedside(AP)
        W = Vector{R}(undef, n)
    else
        n = length(W)
        chkpacked(n, AP)
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(Z)
        Z = Matrix{T}(undef, n, n)
    else
        checksquare(Z) == n || throw(DimensionMismatch("The provided matrix was too small"))
        require_one_based_indexing(Z)
        chkstride1(Z)
    end
    if ismissing(work)
        work = Vector{T}(undef, max(1, 2n -1))
    else
        length(work) < max(1, 2n -1) && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(rwork)
        rwork = Vector{R}(undef, max(1, 3n -2))
    else
        length(rwork) < max(1, 3n -2) && throw(ArgumentError("The provided rwork space was too small"))
        require_one_based_indexing(rwork)
        chkstride1(rwork)
    end
    if PM <: SPMatrixScaled
        AP = packed_unscale!(AP)
        BP = packed_unscale!(BP)
    end
    evs, evecs = spgv!(itype, 'V', uplo, n, APv, BPv, W, Z, max(1, stride(Z, 2)), work, rwork)
    return evs, evecs, BP
end

"""
    spgv!(itype, 'N', uplo, AP::AbstractVector, BP::AbstractVector, W=missing,
        work=missing[, rwork=missing]) -> (W, BP)
    spgv!(itype, 'N', AP::SPMatrix, BP::SPMatrix, W=missing, work=missing
        [, rwork=missing]) -> (W, BP)
    spgv!(itype, 'V', uplo, AP::AbstractVector, BP::AbstractVector, W=missing, Z=missing,
        work=missing[, rwork=missing]) -> (W, Z, BP)
    spgv!(itype, 'V', AP::SPMatrix, BP::SPMatrix, W=missing, Z=missing,
        work=missing[, rwork=missing]) -> (W, Z, BP)

Finds the generalized eigenvalues (second parameter `'N'`) or eigenvalues and eigenvectors (second parameter `'V'`) of a
Hermitian matrix ``A`` in packed storage `AP` and Hermitian positive-definite matrix ``B`` in packed storage `BP`. If
`uplo = 'U'`, `AP` and `BP` are the upper triangles of ``A`` and ``B``, respectively. If `uplo = 'L'`, they are the lower
triangles.

- If `itype = 1`, the problem to solve is ``A x = \\lambda B x``.
- If `itype = 2`, the problem to solve is ``A B x = \\lambda x``.
- If `itype = 3`, the problem to solve is ``B A x = \\lambda x``.

On exit, ``B`` is overwritten with the triangular factor ``U`` or ``L`` from the Cholesky factorization ``B = U' U`` or
``B = L L'``.

The output vector `W` and matrix `Z`, as well as the temporaries `work` and `rwork` (in the complex case) may be specified to
save allocations.
$warnunscale
"""
spgv!

spgvx!(itype::Integer, jobz::AbstractChar, args...) = spgvx!(itype, Val(Symbol(jobz)), args...)
spgvx!(itype::Integer, jobz::Val, uplo::AbstractChar, AP::PM, BP::PM) where {PM<:AbstractVector} =
    spgvx!(itype, jobz, 'A', uplo, AP, BP, nothing, nothing, nothing, nothing, _realtype(eltype(AP))(-1))
spgvx!(itype::Integer, jobz::Val, AP::PM, BP::PM) where {PM<:SPMatrix} =
    spgvx!(itype, jobz, 'A', AP, BP, nothing, nothing, nothing, nothing, _realtype(eltype(AP))(-1))

@pmalso :unscale function spgvx!(itype::Integer, ::Val{:N}, range::AbstractChar, uplo::AbstractChar, AP::PM{T}, BP::PM{T},
    vl::Union{Nothing,T}, vu::Union{Nothing,T}, il::Union{Nothing,<:Integer}, iu::Union{Nothing,<:Integer},
    abstol::T, W::Union{<:AbstractVector{T},Missing}=missing,
    work::Union{<:AbstractVector{T},Missing}=missing,
    iwork::Union{<:AbstractVector{BlasInt},Missing}=missing) where {T<:BlasReal}
    @nospecialize vl vu il iu
    require_one_based_indexing(APv, BPv)
    chkstride1(APv, BPv)
    length(APv) == length(BPv) || throw(DimensionMismatch("The dimensions of AP and BP must be the same"))
    n = packedside(AP)
    if ismissing(W)
        W = Vector{T}(undef, range == 'I' ? iu - il + 1 : n)
    else
        if range == 'A'
            length(W) ≥ n || throw(DimensionMismatch("W has length $(length(W)), but needs at least $n"))
        elseif range == 'I'
            length(W) ≥ iu - il +1 || throw(DimensionMismatch("W has length $(length(W)), but needs at least $(iu - il +1)"))
        end
        # we cannot check 'V'
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(work)
        work = Vector{T}(undef, 8n)
    else
        length(work) < 8n && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(iwork)
        iwork = Vector{BlasInt}(undef, 5n)
    else
        length(iwork) < 5n && throw(ArgumentError("The provided iwork space was too small"))
        require_one_based_indexing(iwork)
        chkstride1(iwork)
    end
    if PM <: SPMatrixScaled
        AP = packed_unscale!(AP)
        BP = packed_unscale!(BP)
    end
    m, _, _, _, _ = spgvx!(itype, 'N', range, uplo, n, APv, BPv, vl, vu, il, iu, abstol, W, Ptr{T}(C_NULL), 1, work, iwork,
        Ptr{BlasInt}(C_NULL))
    return @view(W[1:m]), BP
end

@pmalso :unscale function spgvx!(itype::Integer, ::Val{:V}, range::AbstractChar, uplo::AbstractChar, AP::PM{T}, BP::PM{T},
    vl::Union{Nothing,T}, vu::Union{Nothing,T}, il::Union{Nothing,<:Integer}, iu::Union{Nothing,<:Integer}, abstol::T,
    W::Union{<:AbstractVector{T},Missing}=missing, Z::Union{<:AbstractMatrix{T},Missing}=missing,
    work::Union{<:AbstractVector{T},Missing}=missing, iwork::Union{<:AbstractVector{BlasInt},Missing}=missing,
    ifail::Union{<:AbstractVector{BlasInt},Missing}=missing) where {T<:BlasReal}
    @nospecialize vl vu il iu
    require_one_based_indexing(APv, BPv)
    chkstride1(APv, BPv)
    length(APv) == length(BPv) || throw(DimensionMismatch("The dimensions of AP and BP must be the same"))
    n = packedside(AP)
    if ismissing(W)
        W = Vector{T}(undef, range == 'I' ? iu - il + 1 : n)
    else
        if range == 'A'
            length(W) ≥ n || throw(DimensionMismatch("W has length $(length(W)), but needs at least $n"))
        elseif range == 'I'
            length(W) ≥ iu - il +1 || throw(DimensionMismatch("W has length $(length(W)), but needs at least $(iu - il +1)"))
        end
        # we cannot check 'V'
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(Z)
        Z = Matrix{T}(undef, n, range == 'I' ? iu - il + 1 : n)
    else
        size(Z, 1) == n || throw(DimensionMismatch("Z has first dimension $(size(Z, 1)), but needs $n"))
        if range == 'A'
            size(Z, 2) ≥ n || throw(DimensionMismatch("Z has second dimension $(size(Z, 2)), but needs $n"))
        elseif range == 'I'
            size(Z, 2) ≥ iu - il +1 ||
                throw(DimensionMismatch("Z has second dimension $(size(Z, 2)), but needs at least $(iu - il +1)"))
        end
        require_one_based_indexing(Z)
        chkstride1(Z)
    end
    if ismissing(work)
        work = Vector{T}(undef, 8n)
    else
        length(work) < 8n && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(iwork)
        iwork = Vector{BlasInt}(undef, 5n)
    else
        length(iwork) < 5n && throw(ArgumentError("The provided iwork space was too small"))
        require_one_based_indexing(iwork)
        chkstride1(iwork)
    end
    if ismissing(ifail)
        ifail = Vector{BlasInt}(undef, n)
    else
        length(ifail) < n && throw(DimensionMismatch("The provided ifail space was too small"))
        require_one_based_indexing(ifail)
        chkstride1(ifail)
    end
    if PM <: SPMatrixScaled
        AP = packed_unscale!(AP)
        BP = packed_unscale!(BP)
    end
    m, _, _, info, _ = spgvx!(itype, 'V', range, uplo, n, APv, BPv, vl, vu, il, iu, abstol, W, Z, max(1, stride(Z, 2)), work,
        iwork, ifail)
    return @view(W[1:m]), @view(Z[:, 1:m]), info, @view(ifail[1:m]), BP
end

@pmalso :unscale function spgvx!(itype::Integer, ::Val{:N}, range::AbstractChar, uplo::AbstractChar, AP::PM{T}, BP::PM{T},
    vl::Union{Nothing,R}, vu::Union{Nothing,R}, il::Union{Nothing,<:Integer}, iu::Union{Nothing,<:Integer}, abstol::R,
    W::Union{<:AbstractVector{R},Missing}=missing, work::Union{<:AbstractVector{T},Missing}=missing,
    rwork::Union{<:AbstractVector{R},Missing}=missing,
    iwork::Union{<:AbstractVector{BlasInt},Missing}=missing) where {R<:BlasReal,T<:Complex{R}}
    @nospecialize vl vu il iu
    require_one_based_indexing(APv, BPv)
    chkstride1(APv, BPv)
    length(APv) == length(BPv) || throw(DimensionMismatch("The dimensions of AP and BP must be the same"))
    n = packedside(AP)
    if ismissing(W)
        W = Vector{R}(undef, range == 'I' ? iu - il + 1 : n)
    else
        if range == 'A'
            length(W) ≥ n || throw(DimensionMismatch("W has length $(length(W)), but needs at least $n"))
        elseif range == 'I'
            length(W) ≥ iu - il +1 || throw(DimensionMismatch("W has length $(length(W)), but needs at least $(iu - il +1)"))
        end
        # we cannot check 'V'
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(work)
        work = Vector{T}(undef, 2n)
    else
        length(work) < 2n && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(rwork)
        rwork = Vector{R}(undef, 7n)
    else
        length(rwork) < 7n && throw(ArgumentError("The provided rwork space was too small"))
        require_one_based_indexing(rwork)
        chkstride1(rwork)
    end
    if ismissing(iwork)
        iwork = Vector{BlasInt}(undef, 5n)
    else
        length(iwork) < 5n && throw(ArgumentError("The provided iwork space was too small"))
        require_one_based_indexing(iwork)
        chkstride1(iwork)
    end
    if PM <: SPMatrixScaled
        AP = packed_unscale!(AP)
        BP = packed_unscale!(BP)
    end
    m, _, _, _, _ = spgvx!(itype, 'N', range, uplo, n, APv, BPv, vl, vu, il, iu, abstol, W, Ptr{T}(C_NULL), 1, work, rwork,
        iwork, Ptr{BlasInt}(C_NULL))
    return @view(W[1:m]), BP
end

@pmalso :unscale function spgvx!(itype::Integer, ::Val{:V}, range::AbstractChar, uplo::AbstractChar, AP::PM{T}, BP::PM{T},
    vl::Union{Nothing,R}, vu::Union{Nothing,R}, il::Union{Nothing,<:Integer}, iu::Union{Nothing,<:Integer}, abstol::R,
    W::Union{<:AbstractVector{R},Missing}=missing, Z::Union{<:AbstractMatrix{T},Missing}=missing,
    work::Union{<:AbstractVector{T},Missing}=missing, rwork::Union{<:AbstractVector{R},Missing}=missing,
    iwork::Union{<:AbstractVector{BlasInt},Missing}=missing,
    ifail::Union{<:AbstractVector{BlasInt},Missing}=missing) where {R<:BlasReal,T<:Complex{R}}
    @nospecialize vl vu il iu
    require_one_based_indexing(APv, BPv)
    chkstride1(APv, BPv)
    length(APv) == length(BPv) || throw(DimensionMismatch("The dimensions of AP and BP must be the same"))
    n = packedside(AP)
    if ismissing(W)
        W = Vector{R}(undef, range == 'I' ? iu - il + 1 : n)
    else
        if range == 'A'
            length(W) ≥ n || throw(DimensionMismatch("W has length $(length(W)), but needs at least $n"))
        elseif range == 'I'
            length(W) ≥ iu - il +1 || throw(DimensionMismatch("W has length $(length(W)), but needs at least $(iu - il +1)"))
        end
        # we cannot check 'V'
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(Z)
        Z = Matrix{T}(undef, n, range == 'I' ? iu - il + 1 : n)
    else
        size(Z, 1) == n || throw(DimensionMismatch("Z has first dimension $(size(Z, 1)), but needs $n"))
        if range == 'A'
            size(Z, 2) ≥ n || throw(DimensionMismatch("Z has second dimension $(size(Z, 2)), but needs $n"))
        elseif range == 'I'
            size(Z, 2) ≥ iu - il +1 ||
                throw(DimensionMismatch("Z has second dimension $(size(Z, 2)), but needs at least $(iu - il +1)"))
        end
        require_one_based_indexing(Z)
        chkstride1(Z)
    end
    if ismissing(work)
        work = Vector{T}(undef, 2n)
    else
        length(work) < 2n && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(rwork)
        rwork = Vector{R}(undef, 7n)
    else
        length(rwork) < 7n && throw(ArgumentError("The provided rwork space was too small"))
        require_one_based_indexing(rwork)
        chkstride1(rwork)
    end
    if ismissing(iwork)
        iwork = Vector{BlasInt}(undef, 5n)
    else
        length(iwork) < 5n && throw(ArgumentError("The provided iwork space was too small"))
        require_one_based_indexing(iwork)
        chkstride1(iwork)
    end
    if ismissing(ifail)
        ifail = Vector{BlasInt}(undef, n)
    else
        length(ifail) < n && throw(DimensionMismatch("The provided ifail space was too small"))
        require_one_based_indexing(ifail)
        chkstride1(ifail)
    end
    if PM <: SPMatrixScaled
        AP = packed_unscale!(AP)
        BP = packed_unscale!(BP)
    end
    m, _, _, info, _ = spgvx!(itype, 'V', range, uplo, n, APv, BPv, vl, vu, il, iu, abstol, W, Z, max(1, stride(Z, 2)), work,
        rwork, iwork, ifail)
    return @view(W[1:m]), @view(Z[:, 1:m]), info, @view(ifail[1:m]), BP
end

"""
    spgvx!(itype, 'N', range, uplo, AP::AbstractVector, BP::AbstractVector, vl, vu,
        il, iu, abstol, W=missing, work=missing[, rwork=missing], iwork=missing)
        -> (view(W), BP)
    spgvx!(itype, 'N', range, AP::SPMatrix, BP::SPMatrix, vl, vu, il, iu, abstol,
        W=missing, work=missing[, rwork=missing], iwork=missing) -> (view(W), BP)
    spgvx!(itype, 'V', range, uplo, AP::AbstractVector, BP::AbstractVector, vl, vu,
        il, iu, abstol, W=missing, Z=missing, work=missing[, rwork=missing],
        iwork=missing, ifail=missing) -> (view(W), view(Z), info, view(ifail), BP)
    spgvx!(itype, 'V', range, AP::SPMatrix, BP::SPMatrix, vl, vu, il, iu, abstol,
        W=missing, Z=missing, work=missing[, rwork=missing], iwork=missing, ifail=missing)
        -> (view(W), view(Z), info, view(ifail), BP)

Finds the generalized eigenvalues (second parameter `'N'`) or eigenvalues and eigenvectors (second parameter `'V'`) of a
Hermitian matrix ``A`` in packed storage `AP` and Hermitian positive-definite matrix ``B`` in packed storage `BP`. If
`uplo = 'U'`, `AP` and `BP` are the upper triangles of ``A`` and ``B``, respectively. If `uplo = 'L'`, they are the lower
triangles. If `range = 'A'`, all the eigenvalues are found. If `range = 'V'`, the eigenvalues in the half-open interval
`(vl, vu]` are found. If `range = 'I'`, the eigenvalues with indices between `il` and `iu` are found. `abstol` can be set as a
tolerance for convergence.

- If `itype = 1`, the problem to solve is ``A x = \\lambda B x``.
- If `itype = 2`, the problem to solve is ``A B x = \\lambda x``.
- If `itype = 3`, the problem to solve is ``B A x = \\lambda x``.

On exit, ``B`` is overwritten with the triangular factor ``U`` or ``L`` from the Cholesky factorization ``B = U' U`` or
``B = L L'``.

The eigenvalues are returned in `W` and the eigenvectors in `Z`. `info` is zero upon success or the number of eigenvalues that
failed to converge; their indices are stored in `ifail`. The resulting views to the original data are sized according to the
number of eigenvalues that fulfilled the bounds given by the range.

The output vector `W`, `ifail` and matrix `Z`, as well as the temporaries `work`, `rwork` (in the complex case), and `iwork`
may be specified to save allocations.
$warnunscale
"""
spgvx!

spgvd!(itype::Integer, jobz::AbstractChar, args...) = spgvd!(itype, Val(Symbol(jobz)), args...)

@pmalso :unscale function spgvd!(itype::Integer, ::Val{:N}, uplo::AbstractChar, AP::PM{T}, BP::PM{T},
    W::Union{<:AbstractVector{T},Missing}=missing, work::Union{<:AbstractVector{T},Missing}=missing,
    iwork::Union{<:AbstractVector{BlasInt},Missing}=missing) where {T<:BlasReal}
    require_one_based_indexing(APv, BPv)
    chkstride1(APv, BPv)
    length(APv) == length(BPv) || throw(DimensionMismatch("The dimensions of AP and BP must be the same"))
    if ismissing(W)
        n = packedside(AP)
        W = Vector{T}(undef, n)
    else
        n = length(W)
        chkpacked(n, AP)
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(work)
        work = Vector{T}(undef, n ≤ 1 ? 1 : 2n)
    else
        length(work) < (n ≤ 1 ? 1 : 2n) && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(iwork)
        iwork = Vector{BlasInt}(undef, 1)
    else
        length(iwork) < 1 && throw(ArgumentError("The provided iwork space was too small"))
        require_one_based_indexing(iwork)
        chkstride1(iwork)
    end
    if PM <: SPMatrixScaled
        AP = packed_unscale!(AP)
        BP = packed_unscale!(BP)
    end
    evs, _ = spgvd!(itype, 'N', uplo, n, APv, BPv, W, Ptr{T}(C_NULL), 1, work, length(work), iwork, length(iwork))
    return evs, BP
end

@pmalso :unscale function spgvd!(itype::Integer, ::Val{:V}, uplo::AbstractChar, AP::PM{T}, BP::PM{T},
    W::Union{<:AbstractVector{T},Missing}=missing, Z::Union{<:AbstractMatrix{T},Missing}=missing,
    work::Union{<:AbstractVector{T},Missing}=missing,
    iwork::Union{<:AbstractVector{BlasInt},Missing}=missing) where {T<:BlasReal}
    require_one_based_indexing(APv, BPv)
    chkstride1(APv, BPv)
    length(APv) == length(BPv) || throw(DimensionMismatch("The dimensions of AP and BP must be the same"))
    if ismissing(W)
        n = packedside(AP)
        W = Vector{T}(undef, n)
    else
        n = length(W)
        chkpacked(n, AP)
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(Z)
        Z = Matrix{T}(undef, n, n)
    else
        checksquare(Z) == n || throw(DimensionMismatch("The provided matrix was too small"))
        require_one_based_indexing(Z)
        chkstride1(Z)
    end
    if ismissing(work)
        work = Vector{T}(undef, n ≤ 1 ? 1 : 2n^2 + 6n +1)
    else
        length(work) < (n ≤ 1 ? 1 : 2n^2 + 6n +1) && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(iwork)
        iwork = Vector{BlasInt}(undef, 3 + 5n)
    else
        length(iwork) < 3 + 5n && throw(ArgumentError("The provided iwork space was too small"))
        require_one_based_indexing(iwork)
        chkstride1(iwork)
    end
    if PM <: SPMatrixScaled
        AP = packed_unscale!(AP)
        BP = packed_unscale!(BP)
    end
    evs, evecs = spgvd!(itype, 'V', uplo, n, APv, BPv, W, Z, max(1, stride(Z, 2)), work, length(work), iwork, length(iwork))
    return evs, evecs, BP
end

@pmalso :unscale function spgvd!(itype::Integer, ::Val{:N}, uplo::AbstractChar, AP::PM{T}, BP::PM{T},
    W::Union{<:AbstractVector{R},Missing}=missing, work::Union{<:AbstractVector{T},Missing}=missing,
    rwork::Union{<:AbstractVector{R},Missing}=missing,
    iwork::Union{<:AbstractVector{BlasInt},Missing}=missing) where {R<:BlasReal,T<:Complex{R}}
    require_one_based_indexing(APv, BPv)
    chkstride1(APv, BPv)
    length(APv) == length(BPv) || throw(DimensionMismatch("The dimensions of AP and BP must be the same"))
    if ismissing(W)
        n = packedside(AP)
        W = Vector{R}(undef, n)
    else
        n = length(W)
        chkpacked(n, AP)
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(work)
        work = Vector{T}(undef, n ≤ 1 ? 1 : n)
    else
        length(work) < (n ≤ 1 ? 1 : n) && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(rwork)
        rwork = Vector{R}(undef, n ≤ 1 ? 1 : n)
    else
        length(rwork) < (n ≤ 1 ? 1 : n) && throw(ArgumentError("The provided rwork space was too small"))
        require_one_based_indexing(rwork)
        chkstride1(rwork)
    end
    if ismissing(iwork)
        iwork = Vector{BlasInt}(undef, 1)
    else
        length(iwork) < 1 && throw(ArgumentError("The provided iwork space was too small"))
        require_one_based_indexing(iwork)
        chkstride1(iwork)
    end
    if PM <: SPMatrixScaled
        AP = packed_unscale!(AP)
        BP = packed_unscale!(BP)
    end
    evs, _ = spgvd!(itype, 'N', uplo, n, APv, BPv, W, Ptr{T}(C_NULL), 1, work, length(work), rwork, length(rwork), iwork,
        length(iwork))
    return evs, BP
end

@pmalso :unscale function spgvd!(itype::Integer, ::Val{:V}, uplo::AbstractChar, AP::PM{T}, BP::PM{T},
    W::Union{<:AbstractVector{R},Missing}=missing, Z::Union{<:AbstractMatrix{T},Missing}=missing,
    work::Union{<:AbstractVector{T},Missing}=missing, rwork::Union{<:AbstractVector{R},Missing}=missing,
    iwork::Union{<:AbstractVector{BlasInt},Missing}=missing) where {R<:BlasReal,T<:Complex{R}}
    require_one_based_indexing(APv, BPv)
    chkstride1(APv, BPv)
    length(APv) == length(BPv) || throw(DimensionMismatch("The dimensions of AP and BP must be the same"))
    if ismissing(W)
        n = packedside(AP)
        W = Vector{R}(undef, n)
    else
        n = length(W)
        chkpacked(n, AP)
        require_one_based_indexing(W)
        chkstride1(W)
    end
    if ismissing(Z)
        Z = Matrix{T}(undef, n, n)
    else
        checksquare(Z) == n || throw(DimensionMismatch("The provided matrix was too small"))
        require_one_based_indexing(Z)
        chkstride1(Z)
    end
    if ismissing(work)
        work = Vector{T}(undef, n ≤ 1 ? 1 : 2n)
    else
        length(work) < (n ≤ 1 ? 1 : 2n) && throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    if ismissing(rwork)
        rwork = Vector{R}(undef, n ≤ 1 ? 1 : 2n^2 + 5n +1)
    else
        length(rwork) < (n ≤ 1 ? 1 : 2n^2 + 5n +1) && throw(ArgumentError("The provided rwork space was too small"))
        require_one_based_indexing(rwork)
        chkstride1(rwork)
    end
    if ismissing(iwork)
        iwork = Vector{BlasInt}(undef, 3 + 5n)
    else
        length(iwork) < 3 + 5n && throw(ArgumentError("The provided iwork space was too small"))
        require_one_based_indexing(iwork)
        chkstride1(iwork)
    end
    if PM <: SPMatrixScaled
        AP = packed_unscale!(AP)
        BP = packed_unscale!(BP)
    end
    evs, evecs = spgvd!(itype, 'V', uplo, n, APv, BPv, W, Z, max(1, stride(Z, 2)), work, length(work), rwork, length(rwork),
        iwork, length(iwork))
    return evs, evecs, BP
end

"""
    spgvd!(itype, 'N', uplo, AP::AbstractVector, BP::AbstractVector, W=missing,
        work=missing[, rwork=missing], iwork=missing) -> (W, BP)
    spgvd!(itype, 'N', AP::SPMatrix, BP::SPMatrix, W=missing, work=missing
        [, rwork=missing], iwork=missing) -> (W, BP)
    spgvd!(itype, 'V', uplo, AP::AbstractVector, BP::AbstractVector, W=missing, Z=missing,
        work=missing[, rwork=missing], iwork=missing) -> (W, Z, BP)
    spgvd!(itype, 'V', AP::SPMatrix, BP::SPMatrix, W=missing, Z=missing,
        work=missing[, rwork=missing], iwork=missing) -> (W, Z, BP)

Finds the generalized eigenvalues (second parameter `'N'`) or eigenvalues and eigenvectors (second parameter `'V'`) of a
Hermitian matrix ``A`` in packed storage `AP` and Hermitian positive-definite matrix ``B`` in packed storage `BP`.
If `uplo = 'U'`, `AP` and `BP` are the upper triangles of ``A``, and ``B``, respectively. If `uplo = 'L'`, they are the lower
triangles.

- If `itype = 1`, the problem to solve is ``A x = \\lambda B x``.
- If `itype = 2`, the problem to solve is ``A B x = \\lambda x``.
- If `itype = 3`, the problem to solve is ``B A x = \\lambda x``.

On exit, ``B`` is overwritten with the triangular factor ``U`` or ``L`` from the Cholesky factorization ``B = U' U`` or
``B = L L'``.

The output vector `W` and matrix `Z`, as well as the temporaries `work`, `rwork` (in the complex case), and `iwork` may be
specified to save allocations.

If eigenvectors are desired, it uses a divide and conquer algorithm.
$warnunscale
"""
spgvd!

@pmalso :unscale function hptrd!(uplo::AbstractChar, AP::PM{T}, D::Union{<:AbstractVector{R},Missing}=missing,
    E::Union{<:AbstractVector{R},Missing}=missing,
    τ::Union{<:AbstractVector{T},Missing}=missing) where {R<:BlasReal,T<:Complex{R}}
    require_one_based_indexing(APv)
    chkstride1(APv)
    if ismissing(D)
        n = packedside(AP)
        D = Vector{R}(undef, n)
    else
        n = length(D)
        chkpacked(n, AP)
        require_one_based_indexing(D)
        chkstride1(D)
    end
    if ismissing(E)
        E = Vector{R}(undef, n -1)
    else
        length(E) == n -1 || throw(DimensionMismatch("E has length $(length(E)), but needs $(n-1)"))
        require_one_based_indexing(E)
        chkstride1(E)
    end
    if ismissing(τ)
        τ = Vector{R}(undef, n -1)
    else
        length(τ) == n -1 || throw(DimensionMismatch("τ has length $(length(τ)), but needs $(n-1)"))
        require_one_based_indexing(τ)
        chkstride1(τ)
    end
    PM <: SPMatrixScaled && (AP = packed_unscale!(AP))
    hptrd!(uplo, n, APv, D, E, τ)
    return AP, τ, D, E
end

"""
    hptrd!(uplo, AP::AbstractVector, D=missing, E=missing, τ=missing) -> (A, τ, D, E)
    hptrd!(AP::SPMatrix, D=missing, E=missing, τ=missing) -> (A, τ, D, E)

`hptrd!` reduces a Hermitian matrix ``A`` stored in packed form `AP` to real-symmetric tridiagonal form ``T`` by an orthogonal
similarity transformation: ``Q' A Q = T``.
If `uplo = 'U'`, `AP` is the upper triangle of ``A``, else it is the lower triangle.

On exit, if `uplo = 'U'`, the diagonal and first superdiagonal of ``A`` are overwritten by the corresponding elements of the
tridiagonal matrix ``T``, and the elements above the first superdiagonal, with the array `τ`, represent the orthogonal matrix
``Q`` as a product of elementary reflectors. If `uplo = 'L'` the diagonal and first subdiagonal of ``A`` are over-written by
the corresponding elements of the tridiagonal matrix ``T``, and the elements below the first subdiagonal, with the array `τ`,
represent the orthogonal matrix ``Q`` as a product of elementary reflectors.

`tau` contains the scalar factors of the elementary reflectors, `D` contains the diagonal elements of ``T`` and `E` contains
the off-diagonal elements of ``T``.

The output vectors `D`, `E`, and `τ` may be specified to save allocations.
$warnunscale
"""
hptrd!

@pmalso function opgtr!(uplo::AbstractChar, AP::PM{T}, τ::AbstractVector{T}, Q::Union{<:AbstractMatrix{T},Missing}=missing,
    work::Union{<:AbstractVector{T},Missing}=missing) where {T<:BlasFloat}
    require_one_based_indexing(APv, τ)
    chkstride1(APv, τ)
    n = length(τ) +1
    chkpacked(n, AP)
    if ismissing(Q)
        Q = Matrix{T}(undef, n, n)
    else
        checksquare(Q) == n || throw(DimensionMismatch("Q has size $(size(Q)), but needs ($n, $n)"))
        require_one_based_indexing(Q)
        chkstride1(Q)
    end
    if ismissing(work)
        work = Vector{T}(undef, n -1)
    else
        length(work) ≥ n -1 || throw(ArgumentError("The provided work space was too small"))
        require_one_based_indexing(work)
        chkstride1(work)
    end
    return opgtr!(uplo, n, APv, τ, Q, max(1, stride(Q, 2)), work)
end

"""
    opgtr!(uplo, AP::AbstractVector, τ, Q=missing, work=missing)
    opgtr!(AP::SPMatrixUnscaled, τ, Q=missing, work=missing)

Explicitly finds `Q`, the orthogonal/unitary matrix from [`hptrd!`](@ref). `uplo`, `AP`, and `τ` must correspond to the
input/output to `hptrd!`.

The output matrix `Q` and the temporary `work` may be specified to save allocations.
"""
opgtr!

@pmalso function opmtr!(side::AbstractChar, uplo::AbstractChar, trans::AbstractChar, AP::PM{T}, τ::AbstractVector{T},
    C::AbstractMatrix{T}, work::Union{<:AbstractVector{T},Missing}=missing) where {T<:BlasFloat}
    require_one_based_indexing(APv, τ, C)
    chkstride1(APv, τ, C)
    m, n = size(C)
    if side == 'L'
        chkpacked(m, AP)
        length(τ) == m -1 || throw(DimensionMismatch("τ has length $(length(τ)), needs $(m -1)"))
        if ismissing(work)
            work = Vector{T}(undef, n)
        else
            length(work) ≥ n || throw(ArgumentError("The provided work space was too small"))
            require_one_based_indexing(work)
            chkstride1(work)
        end
    else
        chkpacked(n, AP)
        length(τ) == n -1 || throw(DimensionMismatch("τ has length $(length(τ)), needs $(n -1)"))
        if ismissing(work)
            work = Vector{T}(undef, m)
        else
            length(work) ≥ m || throw(ArgumentError("The provided work space was too small"))
            require_one_based_indexing(work)
            chkstride1(work)
        end
    end
    return opmtr!(side, uplo, trans, m, n, APv, τ, C, max(1, stride(C, 2)), work)
end

@doc raw"""
    opmtr!(side, uplo, trans, AP::AbstractVector, τ, C, work=missing)
    opmtr!(side, trans, AP::SPMatrixUnscaled, τ, C, work=missing)

`opmtr!` overwrites the general ``m \times n`` matrix `C` with

|                                       | `SIDE = 'L'` | `SIDE = 'R'` |
| ---:                                  | :---:        | :---:        |
| `TRANS = 'N'`                         | ``Q C``      | ``C Q``      |
| `TRANS = 'T'` or `'C'` (complex case) | ``Q' C``     | ``C Q'``     |

where ``Q`` is a real orthogonal or complex unitary matrix of order ``n_q``, with ``n_q = m`` if `SIDE = 'L'` and ``n_q = n``
if `SIDE = 'R'`. ``Q`` is defined as the product of ``n_q -1`` elementary reflectors, as returned by [`hptrd!`](@ref) using
packed storage:
- if `UPLO = 'U'`, ``Q = H_{n_q -1} \dotsm H_2 H_1``;
- if `UPLO = 'L'`, ``Q = H_1 H_2 \dotsm H_{n_q -1}``.

The temporary `work` may be specified to save allocations.
"""
opmtr!