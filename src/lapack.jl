export spev!, spevd!, spevx!, pptrf!, spmv!, spr!, tpttr!, trttp!, gemmt!

for (spev, spevd, spevx, pptrf, spmv, spr, tpttr, trttp, gemmt, elty) in
    ((:dspev_,:dspevd_,:dspevx_,:dpptrf_,:dspmv_,:dspr_,:dtpttr_,:dtrttp_,:dgemmt_,:Float64),
     (:sspev_,:sspevd_,:sspevx_,:spptrf_,:sspmv_,:sspr_,:stpttr_,:strttp_,:sgemmt_,:Float32))
     @eval begin
        #       SUBROUTINE DSPEV( JOBZ, UPLO, N, AP, W, Z, LDZ, WORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBZ, UPLO
        #       INTEGER            INFO, LDZ, N
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   AP( * ), W( * ), WORK( * ), Z( LDZ, * )
        function spev!(jobz::AbstractChar, uplo::AbstractChar, n::Integer, AP::AbstractVector{$elty},
                       W::AbstractVector{$elty}=similar(AP, $elty, n),
                       Z::AbstractMatrix{$elty}=similar(AP, $elty, ldz, jobz == 'N' ? 0 : n),
                       work::AbstractVector{$elty}=Vector{$elty}(undef, 3n))
            BLAS.chkuplo(uplo)
            LinearAlgebra.chkstride1(AP, W, Z, work)
            chkpacked(n, AP)
            length(W) ≥ n || throw(ArgumentError("The provided vector was too small"))
            jobz == 'N' || (size(Z, 1) ≥ n && size(Z, 2) ≥ n) || throw(ArgumentError("The provided matrix was too small"))
            ldz = stride(Z, 2)
            ldz ≥ max(1, n) || throw(ArgumentError("The provided matrix had an invalid leading dimension"))
            length(work) < 3n && throw(ArgumentError("The provided work space was too small"))
            info  = Ref{BLAS.BlasInt}()
            ccall((BLAS.@blasfunc($spev), BLAS.libblastrampoline), Cvoid,
                  (Ref{UInt8},        # JOBZ
                   Ref{UInt8},        # UPLO
                   Ref{BLAS.BlasInt}, # N
                   Ptr{$elty},        # AP
                   Ptr{$elty},        # W
                   Ptr{$elty},        # Z
                   Ref{BLAS.BlasInt}, # LDZ
                   Ptr{$elty},        # WORK
                   Ptr{BLAS.BlasInt}, # INFO
                   Clong,             # length(JOBZ)
                   Clong),            # length(UPLO)
                jobz, uplo, n, AP, W, Z, ldz, work, info, 1, 1)
            LAPACK.chklapackerror(info[])
            jobz == 'V' ? (W, Z) : W
        end

        #       SUBROUTINE DSPEVX( JOBZ, RANGE, UPLO, N, AP, VL, VU, IL, IU,
        #      $                   ABSTOL, M, W, Z, LDZ, WORK, IWORK, IFAIL, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBZ, RANGE, UPLO
        #       INTEGER            IL, INFO, IU, LDZ, M, N
        #       DOUBLE PRECISION   ABSTOL, VL, VU
        # *     ..
        # *     .. Array Arguments ..
        #       INTEGER            IFAIL( * ), IWORK( * )
        #       DOUBLE PRECISION   AP( * ), W( * ), WORK( * ), Z( LDZ, * )
        function spevx!(jobz::AbstractChar, range::AbstractChar, uplo::AbstractChar, n::Integer, AP::AbstractVector{$elty},
                        vl::AbstractFloat, vu::AbstractFloat, il::Integer, iu::Integer, abstol::AbstractFloat,
                        W::AbstractVector{$elty}=similar(AP, $elty, n),
                        Z::AbstractMatrix{$elty}=similar(AP, $elty, n, jobz == 'N' ? 0 : (range == 'I' ? iu-il+1 : n)),
                        work::AbstractVector{$elty}=Vector{$elty}(undef, 8n),
                        iwork::AbstractVector{BLAS.BlasInt}=Vector{BLAS.BlasInt}(undef, 5n),
                        ifail::AbstractVector{BLAS.BlasInt}=Vector{BLAS.BlasInt}(undef, jobz == 'V' ? n : 0))
            BLAS.chkuplo(uplo)
            LinearAlgebra.chkstride1(AP, W, Z, work, iwork, ifail)
            chkpacked(n, AP)
            if range == 'I' && !(1 <= il <= iu <= n)
                throw(ArgumentError("illegal choice of eigenvalue indices (il = $il, iu = $iu), which must be between 1 and n = $n"))
            end
            if range == 'V' && vl >= vu
                throw(ArgumentError("lower boundary, $vl, must be less than upper boundary, $vu"))
            end
            length(W) ≥ n || throw(ArgumentError("The provided vector was too small"))
            jobz == 'N' || (size(Z, 1) ≥ n &&
                ((range == 'I' && size(Z, 2) ≥ iu-il+1) ||
                    (range != 'I' && size(Z, 2) ≥ n))) || throw(ArgumentError("The provided matrix was too small"))
            ldz = stride(Z, 2)
            ldz ≥ max(1, n) || throw(ArgumentError("The provided matrix had an invalid leading dimension"))
            length(work) < 8n && throw(ArgumentError("The provided work space was too small"))
            length(iwork) < 5n && throw(ArgumentError("The provided iwork space was too small"))
            jobz == 'V' && length(ifail) < n && throw(ArgumentError("The provided ifail space was too small"))
            m = Ref{BLAS.BlasInt}()
            info = Ref{BLAS.BlasInt}()
            ccall((BLAS.@blasfunc($spevx), BLAS.libblastrampoline), Cvoid,
                  (Ref{UInt8},        # JOBZ
                   Ref{UInt8},        # RANGE
                   Ref{UInt8},        # UPLO
                   Ref{BLAS.BlasInt}, # N
                   Ptr{$elty},        # AP
                   Ref{$elty},        # VL
                   Ref{$elty},        # VU
                   Ref{BLAS.BlasInt}, # IL
                   Ref{BLAS.BlasInt}, # IU
                   Ref{$elty},        # ABSTOL
                   Ptr{BLAS.BlasInt}, # M
                   Ptr{$elty},        # W
                   Ptr{$elty},        # Z
                   Ref{BLAS.BlasInt}, # LDZ
                   Ptr{$elty},        # WORK
                   Ptr{BLAS.BlasInt}, # IWORK
                   Ptr{BLAS.BlasInt}, # IFAIL
                   Ptr{BLAS.BlasInt}, # INFO
                   Clong,             # length(JOBZ)
                   Clong,             # length(RANGE)
                   Clong),            # length(UPLO)
                jobz, range, uplo, n, AP, vl, vu, il, iu, abstol, m, W, Z, ldz, work, iwork, ifail, info, 1, 1, 1)
            LAPACK.chklapackerror(info[])
            @view(W[1:m[]]), @view(Z[:, 1:(jobz == 'V' ? m[] : 0)])
            # We return a view with all rows although Z it might be larger. But if we only go to 1:n, this changes the type of
            # the view, so we are no longer type-stable...
        end
        spevx!(jobz::AbstractChar, n::Integer, A::AbstractVector{$elty}) =
            spevx!(jobz, 'A', 'U', n, A, 0.0, 0.0, 0, 0, -1.0)

        #           SUBROUTINE DSPEVD( JOBZ, UPLO, N, AP, W, Z, LDZ, WORK,
        #      $                   LWORK, IWORK, LIWORK, INFO )
        # *     .. Scalar Arguments ..
        # CHARACTER          JOBZ, UPLO
        # INTEGER            INFO, LDZ, LIWORK, LWORK, N
        # *     ..
        # *     .. Array Arguments ..
        # INTEGER            IWORK( * )
        # DOUBLE PRECISION   AP( * ), W( * ), WORK( * ), Z( LDZ, * )
        function spevd!(jobz::AbstractChar, uplo::AbstractChar, n::Integer, AP::AbstractVector{$elty},
                        W::AbstractVector{$elty}=similar(AP, $elty, n),
                        Z::AbstractMatrix{$elty}=similar(AP, $elty, n, jobz == 'N' ? 0 : n),
                        work::AbstractVector{$elty}=Vector{$elty}(undef, n ≤ 1 ? 1 : (jobz == 'N' ? 2n : 1 + 6n + n^2)),
                        iwork::AbstractVector{BLAS.BlasInt}=Vector{BLAS.BlasInt}(undef, jobz == 'N' || n ≤ 1 ? 1 : 3 + 5n))
            BLAS.chkuplo(uplo)
            LinearAlgebra.chkstride1(AP, W, Z, work, iwork)
            chkpacked(n, AP)
            length(W) ≥ n || throw(ArgumentError("The provided vector was too small"))
            jobz == 'N' || size(Z, 2) ≥ n || throw(ArgumentError("The provided matrix was too small"))
            lwork = BLAS.BlasInt(length(work))
            lwork ≥ (n ≤ 1 ? 1 : (jobz == 'N' ? 2n : 1 + 6n + n^2)) ||
            throw(ArgumentError("The provided work space was too small"))
            liwork = BLAS.BlasInt(length(iwork))
            liwork ≥ (jobz == 'N' || n ≤ 1 ? 1 : 3 + 5n) || throw(ArgumentError("The provided iwork space was too small"))
            ldz = stride(Z, 2)
            ldz ≥ max(1, n) || throw(ArgumentError("The provided matrix had an invalid leading dimension"))
            info = Ref{BLAS.BlasInt}()
            ccall((BLAS.@blasfunc($spevd), BLAS.libblastrampoline), Cvoid,
                   (Ref{UInt8},        # JOBZ
                    Ref{UInt8},        # UPLO
                    Ref{BLAS.BlasInt}, # N
                    Ptr{$elty},        # AP
                    Ptr{$elty},        # W
                    Ptr{$elty},        # Z
                    Ref{BLAS.BlasInt}, # LDZ
                    Ptr{$elty},        # WORK
                    Ref{BLAS.BlasInt}, # LWORK
                    Ptr{BLAS.BlasInt}, # IWORK
                    Ref{BLAS.BlasInt}, # LIWORK
                    Ptr{BLAS.BlasInt}, # INFO
                    Clong,             # length(JOBZ)
                    Clong),            # length(UPLO)
                    jobz, uplo, n, AP, W, Z, ldz, work, lwork, iwork, liwork, info, 1, 1)
            LAPACK.chklapackerror(info[])
            jobz == 'V' ? (W, Z) : W
        end

        # SUBROUTINE DPPTRF( UPLO, N, AP, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          UPLO
        #       INTEGER            INFO, N
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   AP( * )
        function pptrf!(uplo::AbstractChar, n::Integer, AP::AbstractVector{$elty})
            LinearAlgebra.require_one_based_indexing(AP)
            LinearAlgebra.chkstride1(AP)
            BLAS.chkuplo(uplo)
            chkpacked(n, AP)
            info = Ref{BLAS.BlasInt}()
            ccall((BLAS.@blasfunc($pptrf), BLAS.libblastrampoline), Cvoid,
                  (Ref{UInt8},        # UPLO
                   Ref{BLAS.BlasInt}, # N
                   Ptr{$elty},        # AP
                   Ptr{BLAS.BlasInt}, # INFO
                   Clong),            # length(UPLO)
                  uplo, n, AP, info, 1)
            LAPACK.chkargsok(info[])
            #info[] > 0 means the leading minor of order info[] is not positive definite
            #ordinarily, throw Exception here, but return error code here
            #this simplifies isposdef! and factorize
            return AP, info[] # info stored in Cholesky
        end

        #           SUBROUTINE DSPMV( UPLO, N, ALPHA, AP, X, INCX, BETA, Y, INCY )
        # *     .. Scalar Arguments ..
        #         DOUBLE PRECISION ALPHA,BETA
        #         INTEGER INCX,INCY,N
        #         CHARACTER UPLO
        #   *     ..
        #   *     .. Array Arguments ..
        #         DOUBLE PRECISION AP(*),X(*),Y(*)
        function spmv!(uplo::AbstractChar, α::Union{$elty,Bool}, AP::AbstractVector{$elty}, x::AbstractVector{$elty},
                       β::Union{$elty,Bool}, y::AbstractVector{$elty})
            BLAS.chkuplo(uplo)
            LinearAlgebra.chkstride1(AP)
            LinearAlgebra.require_one_based_indexing(AP, x, y)
            n = length(x)
            length(y) == n || ArgumentError("Input and output dimensions must match")
            chkpacked(n, AP)
            px, stx = BLAS.vec_pointer_stride(x, ArgumentError("input vector with 0 stride is not allowed"))
            py, sty = BLAS.vec_pointer_stride(y, ArgumentError("dest vector with 0 stride is not allowed"))
            GC.@preserve x AP ccall((BLAS.@blasfunc($spmv), BLAS.libblastrampoline), Cvoid,
                (Ref{UInt8},        # UPLO
                 Ref{BLAS.BlasInt}, # N
                 Ref{$elty},        # ALPHA
                 Ptr{$elty},        # AP
                 Ptr{$elty},        # X
                 Ref{BLAS.BlasInt}, # INCX
                 Ref{$elty},        # BETA
                 Ptr{$elty},        # Y
                 Ref{BLAS.BlasInt}, # INCY
                 Clong),            # length(UPLO)
                uplo, n, α, AP, px, stx, β, py, sty, 1)
            return y
        end

        #           SUBROUTINE DSPR( UPLO, N, ALPHA, X, INCX, AP )
        # *     .. Scalar Arguments ..
        #       DOUBLE PRECISION ALPHA
        #       INTEGER INCX,N
        #       CHARACTER UPLO
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION AP(*),X(*)
        function spr!(uplo::AbstractChar, α::$elty, x::AbstractVector{$elty}, AP::AbstractVector{$elty})
            BLAS.chkuplo(uplo)
            LinearAlgebra.chkstride1(AP)
            LinearAlgebra.require_one_based_indexing(AP, x)
            n = length(x)
            chkpacked(n, AP)
            px, stx = BLAS.vec_pointer_stride(x, ArgumentError("input vector with 0 stride is not allowed"))
            GC.@preserve x ccall((BLAS.@blasfunc($spr), BLAS.libblastrampoline), Cvoid,
                (Ref{UInt8},        # UPLO
                 Ref{BLAS.BlasInt}, # N
                 Ref{$elty},        # ALPHA
                 Ptr{$elty},        # X
                 Ref{BLAS.BlasInt}, # INCX
                 Ptr{$elty},        # AP
                 Clong),            # length(UPLO)
                uplo, n, α, px, stx, AP, 1)
            return AP
        end

        #           SUBROUTINE DTPTTR( UPLO, N, AP, A, LDA, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          UPLO
        #       INTEGER            INFO, N, LDA
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   A( LDA, * ), AP( * )
        function tpttr!(uplo::AbstractChar, AP::AbstractVector{$elty}, A::StridedArray{$elty})
            BLAS.chkuplo(uplo)
            LinearAlgebra.chkstride1(AP)
            LinearAlgebra.require_one_based_indexing(AP, A)
            n = LinearAlgebra.checksquare(A)
            chkpacked(n, AP)
            lda = Base.stride(A, 2)
            info = Ref{BLAS.BlasInt}()
            ccall((BLAS.@blasfunc($tpttr), BLAS.libblastrampoline), Cvoid,
                (Ref{UInt8},        # UPLO
                 Ref{BLAS.BlasInt}, # N
                 Ptr{$elty},        # AP
                 Ptr{$elty},        # A
                 Ref{BLAS.BlasInt}, # LDA
                 Ptr{BLAS.BlasInt}, # INFO
                 Clong),            # length(UPLO)
                uplo, n, AP, A, lda, info, 1)
            LAPACK.chklapackerror(info[])
            return A
        end

        #           SUBROUTINE DTRTTP( UPLO, N, A, LDA, AP, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          UPLO
        #       INTEGER            INFO, N, LDA
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   A( LDA, * ), AP( * )
        function trttp!(uplo::AbstractChar, A::StridedArray{$elty}, AP::AbstractVector{$elty})
            BLAS.chkuplo(uplo)
            LinearAlgebra.chkstride1(AP)
            LinearAlgebra.require_one_based_indexing(AP, A)
            n = LinearAlgebra.checksquare(A)
            chkpacked(n, AP)
            lda = Base.stride(A, 2)
            info = Ref{BLAS.BlasInt}()
            ccall((BLAS.@blasfunc($trttp), BLAS.libblastrampoline), Cvoid,
                (Ref{UInt8},        # UPLO
                 Ref{BLAS.BlasInt}, # N
                 Ptr{$elty},        # A
                 Ref{BLAS.BlasInt}, # LDA
                 Ptr{$elty},        # AP
                 Ptr{BLAS.BlasInt}, # INFO
                 Clong),            # length(UPLO)
                uplo, n, A, lda, AP, info, 1)
            LAPACK.chklapackerror(info[])
            return AP
        end

        # Not part of the reference BLAS (yet), but present in many implementations, also Julia 1.9
        # Requires at least OpenBlas 0.3.22, but Julia currently maximally ships with 0.3.17. But with this construction, we
        # make the function available anyway - just load MKL before this package. When the function is not found, the more
        # inefficient gemm! is called.
        #           SUBROUTINE DGEMMT( UPLO, TRANSA, TRANSB, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
        #       .. Scalar Arguments ..
        #       DOUBLE PRECISION ALPHA,BETA
        #       INTEGER K,LDA,LDB,LDC,N
        #       CHARACTER TRANSA,TRANSB, UPLO
        #       ..
        #       .. Array Arguments ..
        #       DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
        if BLAS.lbt_get_forward(BLAS.@blasfunc($gemmt),
            BLAS.USE_BLAS64 ? BLAS.LBT_INTERFACE_ILP64 : BLAS.LBT_INTERFACE_LP64) ∈ (BLAS.lbt_get_default_func(), Ptr{Cvoid}(-1))
            function gemmt!(uplo::AbstractChar, transA::AbstractChar, transB::AbstractChar, alpha::Union{($elty), Bool},
                A::AbstractVecOrMat{$elty}, B::AbstractVecOrMat{$elty}, beta::Union{($elty), Bool},
                C::AbstractVecOrMat{$elty})
                return BLAS.gemm!(transA, transB, alpha, A, B, beta, C)
            end
        else
            function gemmt!(uplo::AbstractChar, transA::AbstractChar, transB::AbstractChar, alpha::Union{($elty), Bool},
                            A::AbstractVecOrMat{$elty}, B::AbstractVecOrMat{$elty}, beta::Union{($elty), Bool},
                            C::AbstractVecOrMat{$elty})
                LinearAlgebra.require_one_based_indexing(A, B, C)
                BLAS.chkuplo(uplo)
                m = size(A, transA == 'N' ? 1 : 2)
                ka = size(A, transA == 'N' ? 2 : 1)
                kb = size(B, transB == 'N' ? 1 : 2)
                n = size(B, transB == 'N' ? 2 : 1)
                if ka != kb || m != LinearAlgebra.checksquare(C) || n != m
                    throw(DimensionMismatch(lazy"A has size ($m,$ka), B has size ($kb,$n), C has size $(size(C))"))
                end
                LinearAlgebra.chkstride1(A)
                LinearAlgebra.chkstride1(B)
                LinearAlgebra.chkstride1(C)
                ccall((BLAS.@blasfunc($gemmt), BLAS.libblastrampoline), Cvoid,
                    (Ref{UInt8},        # UPLO
                    Ref{UInt8},        # TRANSA
                    Ref{UInt8},        # TRANSB
                    Ref{BLAS.BlasInt}, # N
                    Ref{BLAS.BlasInt}, # K
                    Ref{$elty},        # ALPHA
                    Ptr{$elty},        # A
                    Ref{BLAS.BlasInt}, # LDA
                    Ptr{$elty},         # B
                    Ref{BLAS.BlasInt}, # LDB
                    Ref{$elty},        # BETA
                    Ptr{$elty},        # C
                    Ref{BLAS.BlasInt}, # LDC
                    Clong,             # length(UPLO)
                    Clong,             # length(TRANSA)
                    Clong,             # length(TRANSB)
                    ),
                    uplo, transA, transB, n, ka, alpha, A, max(1, Base.stride(A, 2)), B, max(1, Base.stride(B,2)), beta, C,
                    max(1, Base.stride(C, 2)), 1, 1, 1)
                C
            end
        end
    end
end

@doc """
    spev!(jobz, uplo, n, AP; [W, Z, work])

`spev!` computes all the eigenvalues and, optionally, eigenvectors of a real symmetric matrix `A` in packed storage.
""" spev!

@doc """
    spevx!(jobz, range, uplo, n, AP, vl, vu, il, iu, abstol; [W, Z,  work, iwork, ifail])

`spevx!` computes selected eigenvalues and, optionally, eigenvectors of a real symmetric matrix `A` in packed storage.
Eigenvalues/vectors can be selected by specifying either a range of values or a range of indices for the desired eigenvalues.
""" spevx!

@doc """
    spevd!(jobz, uplo, n, AP; [W, Z, work, iwork])

`spevd!` computes all the eigenvalues and, optionally, eigenvectors of a real symmetric matrix `A` in packed storage. If
eigenvectors are desired, it uses a divide and conquer algorithm.
""" spevd!

@doc raw"""
    pptrf!(uplo, n, AP)

`pptrf!` computes the Cholesky factorization of a real symmetric positive definite matrix `A` stored in packed format.

The factorization has the form
- ``A = U^\top U``,  if `uplo == 'U'`, or
- ``A = L L^\top``,  if `uplo == 'L'`,
where ``U`` is an upper triangular matrix and ``L`` is lower triangular.
""" pptrf!

@doc raw"""
    spmv!(uplo, α, AP, x, β, y)

`spmv!` performs the matrix-vector operation
``y := \alpha A x + \beta y``,

where `α` and `β` are scalars, `x` and `y` are `n` element vectors and `A` is an `n × n` symmetric matrix, supplied in packed
form.
""" spmv!

@doc raw"""
    spr!(uplo, α, x, AP)

`spr!` performs the symmetric rank 1 operation
``A := \alpha x x^\top + A``,
where `α` is a real scalar, `x` is an `n` element vector and `A` is an `n × n` symmetric matrix, supplied in packed form.
""" spr!

"""
    tpttr!(uplo, AP, A)

`tpttr!` copies a triangular matrix from the standard packed format (TP) to the standard full format (TR).

!!! info
    This function is also implemented in plain Julia and therefore works with arbitrary element types.
"""
function tpttr!(uplo::AbstractChar, AP::AbstractVector{T}, A::AbstractMatrix{T}) where {T}
    BLAS.chkuplo(uplo)
    n = LinearAlgebra.checksquare(A)
    chkpacked(n, AP)
    if uplo == 'L' || uplo == 'l'
        k = 0
        @inbounds for j in 1:n, i in j:n
            k += 1
            A[i, j] = AP[k]
        end
    else
        k = 0
        @inbounds for j in 1:n, i in 1:j
            k += 1
            A[i, j] = AP[k]
        end
    end
    return A
end

"""
    trttp!(uplo, A, AP)

`trttp!` copies a triangular matrix from the standard full format (TR) to the standard packed format (TP).

!!! info
    This function is also implemented in plain Julia and therefore works with arbitrary element types.
"""
function trttp!(uplo::AbstractChar, A::AbstractMatrix{T}, AP::AbstractVector{T}) where {T}
    BLAS.chkuplo(uplo)
    n = LinearAlgebra.checksquare(A)
    chkpacked(n, AP)
    if uplo == 'L' || uplo == 'l'
        k = 0
        @inbounds for j in 1:n, i in j:n
            k += 1
            AP[k] = A[i, j]
        end
    else
        k = 0
        @inbounds for j in 1:n, i in 1:j
            k += 1
            AP[k] = A[i, j]
        end
    end
    return AP
end

@doc """
    gemmt!(uplo, transA, transB, alpha, A, B, beta, C)

`gemmt!` computes a matrix-matrix product with general matrices but updates only the upper or lower triangular part of the
result matrix.

!!! info
    This function is a recent BLAS extension; for OpenBLAS, it requires at least version 0.3.22 (which is not yet shipped with
    Julia). If the currently available BLAS does not offer `gemmt`, the function falls back to `gemm`.
""" gemmt!