# Symmetric square matrix, where we only store the upper triangle in col-major vectorized form
# The PackedMatrix can be effciently broadcast to other matrices, which works on the vector representation. It can also be
# receive values from generic matrices by copyto! or broadcasting. Using it in a mixed chain of broadcastings of different type
# is not implemented and will potentially lead to logical errors (in the sense that the types will not match) or even segfaults
# (as the correct index mapping is not implemented).
struct PackedMatrix{R,V<:AbstractVector{R}} <: AbstractMatrix{R}
    dim::Int
    data::V

    function PackedMatrix(dim::Integer, data::AbstractVector{R}) where {R}
        chkpacked(dim, data)
        return new{R,typeof(data)}(dim, data)
    end

    function PackedMatrix{R}(::UndefInitializer, dim::Integer) where {R}
        return new{R,Vector{R}}(dim, Vector{R}(undef, packedsize(dim)))
    end
end

@inline packedsize(dim) = dim * (dim +1) ÷ 2
@inline rowcol_to_vec(row, col) = (@boundscheck(@assert(1 ≤ row ≤ col)); return col * (col -1) ÷ 2 + row)
# check whether a linear index corresponds to a diagonal
function isdiag_triu(idx::Signed)
    col = 2
    while idx > 1
        idx -= col
        col += 1
    end
    return idx == 1
end
struct PackedDiagonalIterator
    dim::Int
    k::UInt
    PackedDiagonalIterator(dim, k) = new(dim, abs(k))
end
function Base.iterate(iter::PackedDiagonalIterator)
    if iter.k ≥ iter.dim
        return nothing
    else
        j = iter.k * (iter.k +1) ÷ 2 +1
        return j, (j, iter.k +2)
    end
end
function Base.iterate(iter::PackedDiagonalIterator, state)
    j, δ = state
    j += δ
    if δ > iter.dim
        return nothing
    else
        return j, (j, δ +1)
    end
end
Base.IteratorSize(::Type{PackedDiagonalIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{PackedDiagonalIterator}) = Base.HasEltype()
Base.eltype(::PackedDiagonalIterator) = Int
Base.length(iter::PackedDiagonalIterator) = iter.dim - iter.k
LinearAlgebra.diagind(A::PackedMatrix, k::Integer=0) = collect(PackedDiagonalIterator(A.dim, k))
function LinearAlgebra.diag(A::PackedMatrix{R}, k::Integer=0) where {R}
    iter = PackedDiagonalIterator(A.dim, k)
    diagonal = Vector{R}(undef, length(iter))
    for (i, idx) in enumerate(iter)
        @inbounds diagonal[i] = A[idx]
    end
    return diagonal
end
function LinearAlgebra.tr(A::PackedMatrix{R}) where {R}
    trace = zero(R)
    for idx in PackedDiagonalIterator(A.dim, 0)
        @inbounds trace += A[idx]
    end
    return trace
end

Base.size(P::PackedMatrix) = (P.dim, P.dim)
Base.eltype(::PackedMatrix{R}) where {R} = R
Base.@propagate_inbounds Base.getindex(P::PackedMatrix, idx) = P.data[idx]
Base.@propagate_inbounds Base.getindex(P::PackedMatrix, row, col) =
    P.data[@inbounds rowcol_to_vec(min(row, col), max(row, col))]
Base.@propagate_inbounds Base.setindex!(P::PackedMatrix, X, idx::Union{Integer,LinearIndices}) = P.data[idx] = X
Base.@propagate_inbounds Base.setindex!(P::PackedMatrix, X, row, col) =
    P.data[@inbounds rowcol_to_vec(min(row, col), max(row, col))] = X
Base.IndexStyle(::PackedMatrix) = IndexLinear()
Base.iterate(P::PackedMatrix, args...) = iterate(P.data, args...)
Base.length(P::PackedMatrix) = length(P.data)
Base.fill!(P::PackedMatrix{R}, x::R) where {R} = fill!(P.data, x)
Base.copy(P::PackedMatrix) = PackedMatrix(P.dim, copy(P.data))
for cp in (:copy!, :copyto!)
    @eval begin
        Base.@propagate_inbounds function Base.$cp(dst::PackedMatrix{R}, src::PackedMatrix{R}) where {R}
            $cp(dst.data, src.data)
            return dst
        end
        Base.@propagate_inbounds function Base.$cp(dst::PackedMatrix{R}, src::AbstractVector{R}) where {R}
            $cp(dst.data, src)
            return dst
        end
        Base.@propagate_inbounds Base.$cp(dst::AbstractVector{R}, src::PackedMatrix{R}) where {R} = $cp(dst, src.data)
    end
end
function Base.copyto!(dst::PackedMatrix{R}, src::AbstractMatrix{R}) where {R}
    j = 1
    for (i, col) in enumerate(eachcol(src))
        @inbounds copyto!(dst.data, j, col, 1, i)
        j += i
    end
    return dst
end
function Base.copy!(dst::PackedMatrix{R}, src::AbstractMatrix{R}) where {R}
    @boundscheck checkbounds(src, axes(dst)...)
    return copyto!(dst, src)
end
function Base.similar(P::PackedMatrix{R}, ::Type{T}=eltype(P), dims::NTuple{2,Int}=size(P)) where {R,T}
    ==(dims...) || error("Packed matrices must be square")
    dim = first(dims)
    return PackedMatrix(dim, similar(P.data, T, packedsize(dim)))
end
LinearAlgebra.vec(P::PackedMatrix) = P.data
Base.convert(T::Type{<:Ptr}, P::PackedMatrix) = convert(T, P.data)

# Broadcasting
Base.broadcastable(P::PackedMatrix) = PackedMatrixBroadcasting{typeof(P)}(P)
struct PackedMatrixBroadcasting{PM<:PackedMatrix} <: AbstractVector{eltype(PM)}
    data::PM
end
Base.size(PB::PackedMatrixBroadcasting) = size(PB.data.data)
Base.axes(PB::PackedMatrixBroadcasting) = axes(PB.data.data)
Base.getindex(PB::PackedMatrixBroadcasting, ind::Int) = PB.data.data[ind]
Base.setindex!(PB::PackedMatrixBroadcasting, val, ind::Int) = PB.data.data[ind] = val

struct PackedMatrixStyle <: Broadcast.AbstractArrayStyle{1} end
PackedMatrixStyle(::Val{1}) = PackedMatrixStyle()
PackedMatrixStyle(::Val{2}) = PackedMatrixGenericStyle()
Base.BroadcastStyle(::Type{<:Union{<:PackedMatrixBroadcasting,<:PackedMatrix}}) = PackedMatrixStyle()
Base.similar(bc::Broadcast.Broadcasted{PackedMatrixStyle}, ::Type{T}) where {T} = similar(find_pm(bc).data, T)
find_pm(bc::Base.Broadcast.Broadcasted) = find_pm(bc.args)
find_pm(args::Tuple) = find_pm(find_pm(args[1]), Base.tail(args))
find_pm(x) = x
find_pm(::Tuple{}) = nothing
find_pm(P::PackedMatrixBroadcasting, ::Any) = P
find_pm(::Any, rest) = find_pm(rest)
@inline function Base.copyto!(dest::PackedMatrix, bc::Broadcast.Broadcasted{PackedMatrixStyle})
    axes(dest.data) == axes(bc) || Broadcast.throwdm(axes(dest.data), axes(bc))
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if bc.f === identity && bc.args isa Tuple{AbstractArray} # only a single input argument to broadcast!
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end
    bc′ = Broadcast.preprocess(dest, bc)
    # Performance may vary depending on whether `@inbounds` is placed outside the
    # for loop or not. (cf. https://github.com/JuliaLang/julia/issues/38086)
    @inbounds @simd for I in eachindex(bc′)
        dest[I] = bc′[I]
    end
    return dest
end
Base.Broadcast.materialize!(s::PackedMatrixStyle, dest::PackedMatrix, bc::Broadcast.Broadcasted{PackedMatrixStyle}) =
    (Base.Broadcast.materialize!(s, dest.data, bc); return dest)
struct PackedMatrixGenericStyle <: Broadcast.AbstractArrayStyle{2} end
PackedMatrixGenericStyle(::Val{1}) = PackedMatrixStyle()
PackedMatrixGenericStyle(::Val{2}) = PackedMatrixGenericStyle()
Base.BroadcastStyle(::PackedMatrixStyle, ::Broadcast.DefaultArrayStyle{2}) = PackedMatrixGenericStyle()
Base.BroadcastStyle(::PackedMatrixGenericStyle, ::Broadcast.DefaultArrayStyle{1}) = PackedMatrixStyle()
@inline function Base.copyto!(dest::PackedMatrix, bc::Broadcast.Broadcasted{PackedMatrixGenericStyle})
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    if bc.f === identity && bc.args isa Tuple{AbstractArray}
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end
    bc′ = Broadcast.preprocess(dest, bc)
    @inbounds @simd for I in eachindex(IndexCartesian(), bc′) # linear indices will be out of bounds!
        dest[I] = bc′[I]
    end
    return dest
end

chkpacked(n::Integer, AP::AbstractVector) = 2length(AP) == n * (n +1) || error("Packed storage length does not match dimension")
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
        function spev!(jobz::AbstractChar, uplo::AbstractChar, n::Integer, AP::AbstractVector{$elty})
            BLAS.chkuplo(uplo)
            LinearAlgebra.chkstride1(AP)
            chkpacked(n, AP)
            W     = similar(AP, $elty, n)
            work  = Vector{$elty}(undef, 3n)
            info  = Ref{BLAS.BlasInt}()
            ldz   = n
            if jobz == 'N'
                Z = similar(AP, $elty, ldz, 0)
            else
                Z = similar(AP, $elty, ldz, n)
            end
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
                        vl::AbstractFloat, vu::AbstractFloat, il::Integer, iu::Integer, abstol::AbstractFloat)
            BLAS.chkuplo(uplo)
            LinearAlgebra.chkstride1(AP)
            chkpacked(n, AP)
            if range == 'I' && !(1 <= il <= iu <= n)
                throw(ArgumentError("illegal choice of eigenvalue indices (il = $il, iu = $iu), which must be between 1 and n = $n"))
            end
            if range == 'V' && vl >= vu
                throw(ArgumentError("lower boundary, $vl, must be less than upper boundary, $vu"))
            end
            m = Ref{BLAS.BlasInt}()
            W = similar(AP, $elty, n)
            ldz = n
            if jobz == 'N'
                Z = similar(AP, $elty, ldz, 0)
            elseif jobz == 'V'
                Z = similar(AP, $elty, ldz, n)
            end
            work   = Vector{$elty}(undef, 8n)
            iwork  = Vector{BLAS.BlasInt}(undef, 5n)
            ifail  = Vector{BLAS.BlasInt}(undef, n)
            info   = Ref{BLAS.BlasInt}()
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
                jobz, range, uplo, n, AP, vl, vu, il, iu, abstol, m, W, Z, max(1, ldz), work, iwork, ifail, info, 1, 1, 1)
            LAPACK.chklapackerror(info[])
            W[1:m[]], Z[:,1:(jobz == 'V' ? m[] : 0)]
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
        function spevd!(jobz::AbstractChar, uplo::AbstractChar, n::Integer, AP::AbstractVector{$elty})
            BLAS.chkuplo(uplo)
            LinearAlgebra.chkstride1(AP)
            chkpacked(n, AP)
            W      = similar(AP, $elty, n)
            ldz    = n
            if jobz == 'N'
                Z = similar(AP, $elty, ldz, 0)
            else
                Z = similar(AP, $elty, ldz, n)
            end
            work   = Vector{$elty}(undef, 1)
            lwork  = BLAS.BlasInt(-1)
            iwork  = Vector{LinearAlgebra.BlasInt}(undef, 1)
            liwork = BLAS.BlasInt(-1)
            info   = Ref{BLAS.BlasInt}()
            for i = 1:2  # first call returns lwork as work[1]
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
                if i == 1
                    lwork = BLAS.BlasInt(work[1])
                    resize!(work, lwork)
                    liwork = BLAS.BlasInt(iwork[1])
                    resize!(iwork, liwork)
                end
            end
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

LinearAlgebra.axpy!(a::Number, x::PackedMatrix, y::AbstractVector) = LinearAlgebra.axpy!(a, x.data, y)
LinearAlgebra.axpy!(a::Number, x::AbstractVector, y::PackedMatrix) = LinearAlgebra.axpy!(a, x, y.data)
LinearAlgebra.axpy!(a::Number, x::PackedMatrix, y::PackedMatrix) = LinearAlgebra.axpy!(a, x.data, y.data)
# TODO: double-sparse implementations similar to dot


# These are for multiplications where the matrix is directly interpreted as a vector. Note that this implies that off-diagonal
# elements only enter once.
LinearAlgebra.mul!(C::PackedMatrix, A::AbstractMatrix, B::AbstractVector, α::Number, β::Number) = mul!(C.data, A, B, α, β)
LinearAlgebra.mul!(C::AbstractVector, A::AbstractMatrix, B::PackedMatrix, α::Number, β::Number) = mul!(C, A, B.data, α, β)
# And this is the symmetric multiplication
LinearAlgebra.mul!(C::AbstractVector, A::PackedMatrix, B::AbstractVector, α::Number, β::Number) =
    spmv!('U', α, A.data, B, β, C)
# Also a wrapper for rank-one updates
spr!(α, x::AbstractVector, AP::PackedMatrix) = spr!('U', α, x, AP.data)
function Base.Matrix{T}(A::PackedMatrix{T}) where {T}
    result = Matrix{T}(undef, A.dim, A.dim)
    tpttr!('U', A.data, result)
    return Symmetric(result)
end
function PackedMatrix(A::Symmetric{T,<:AbstractMatrix{T}}) where {T}
    A.uplo == 'U' || error("PackedMatrix requires the input matrix to be stored in the upper triangle.")
    result = PackedMatrix{T}(undef, size(A, 1))
    trttp!('U', parent(A), result.data)
    return result
end

# TODO: improve by replacing isdiag_triu with running next diagonal index
function LinearAlgebra.dot(A::PackedMatrix{R}, B::PackedMatrix{R}) where {R}
    A.dim == B.dim || error("Matrices must have same dimensions")
    result = zero(R)
    i = 1
    @inbounds @simd for j in 1:A.dim
        for _ in 1:j-1
            result += 2conj(A[i]) * B[i]
            i += 1
        end
        result += conj(A[i]) * B[i]
        i += 1
    end
    return result
end
function LinearAlgebra.dot(A::PackedMatrix{R,<:SparseVector}, B::PackedMatrix{R}) where {R}
    A.dim == B.dim || error("Matrices must have same dimensions")
    result = zero(R)
    nzs = rowvals(A.data)
    vs = nonzeros(A.data)
    @inbounds @simd for i in 1:length(nzs) # for @simd, cannot iterate over (j, v)
        j = nzs[i]
        v = vs[i]
        if isdiag_triu(j)
            result += conj(v) * B[j]
        else
            result += 2conj(v) * B[j]
        end
    end
    return result
end
LinearAlgebra.dot(A::PackedMatrix{R}, B::PackedMatrix{R,<:SparseVector}) where {R} = conj(dot(B, A))
function LinearAlgebra.dot(A::PackedMatrix{R,<:SparseVector}, B::PackedMatrix{R,<:SparseVector}) where {R}
    A.dim == B.dim || error("Matrices must have same dimensions")
    result = zero(R)
    nzA = rowvals(A.data)
    nzB = rowvals(B.data)
    vA = nonzeros(A.data)
    vB = nonzeros(B.data)
    iA = length(nzA)
    iB = length(nzB)
    @inbounds while iA ≥ 1 && iB ≥ 1
        pA = nzA[iA]
        pB = nzB[iB]
        if pA == pB
            if isdiag_triu(pA)
                result += conj(vA[pA]) * vB[pA]
            else
                result += 2conj(vA[pA]) * pB[pA]
            end
        else
            while pA < pB
                iB -= 1
                pB = nzB[iB]
            end
            while pB < pA
                iA -= 1
                pA = nzA[iA]
            end
        end
    end
    return result
end

function LinearAlgebra.norm(pm::PackedMatrix{R}, p::Real=2) where {R}
    result = zero(R)
    i = 1
    @inbounds @simd for j in 1:pm.dim
        for _ in 1:j-1
            result += 2abs(pm.data[i])^p
            i += 1
        end
        result += abs(pm.data[i])^p
        i += 1
    end
    return result^(1/p)
end

function LinearAlgebra.norm2(pm::PackedMatrix{R}) where {R<:Real}
    result = R(2) * BLAS.nrm2(pm.data)^2
    i = 0
    @inbounds @simd for j in 1:pm.dim
        i += j
        result -= pm.data[i]^2
    end
    return sqrt(result)
end
function LinearAlgebra.norm2(pm::PackedMatrix{R,<:SparseVector}) where {R<:Real}
    nzs = rowvals(pm.data)
    vs = nonzeros(pm.data)
    result = R(2) * BLAS.nrm2(vs)^2
    @inbounds @simd for i in 1:length(nzs)
        j = nzs[i]
        if isdiag_triu(j)
            result -= abs(vs[i])^2
        end
    end
    return sqrt(result)
end
LinearAlgebra.eigen!(pm::PackedMatrix{R}) where {R<:Real} = Eigen(spevd!('V', 'U', pm.dim, pm.data)...)
LinearAlgebra.eigvals(pm::PackedMatrix{R}) where {R<:Real} = spevd!('N', 'U', pm.dim, copy(pm.data))
LinearAlgebra.eigen!(pm::PackedMatrix{R}, vl::R, vu::R) where {R<:Real} =
    Eigen(spevx!('V', 'V', 'U', pm.dim, pm.data, vl, vu, 0, 0, -one(R))...)
LinearAlgebra.eigen!(pm::PackedMatrix{R}, range::UnitRange) where {R<:Real} =
    Eigen(spevx!('V', 'I', 'U', pm.dim, pm.data, zero(R), zero(R), range.start, range.stop, -one(R))...)
LinearAlgebra.eigvals!(pm::PackedMatrix{R}, vl::R, vu::R) where {R<:Real} =
    spevx!('N', 'V', 'U', pm.dim, pm.data, vl, vu, 0, 0, -one(R))[1]
LinearAlgebra.eigvals!(pm::PackedMatrix{R}, range::UnitRange) where {R<:Real} =
    spevx!('N', 'I', 'U', pm.dim, pm.data, zero(R), zero(R), range.start, range.stop, -one(R))[1]
eigmin!(pm::PackedMatrix) = eigvals!(pm, 1:1)[1]
eigmax!(pm::PackedMatrix) = eigvals!(pm, pm.dim:pm.dim)[1]
LinearAlgebra.eigmin(pm::PackedMatrix) = eigmin!(copy(pm))
LinearAlgebra.eigmax(pm::PackedMatrix) = eigmax!(copy(pm))
function LinearAlgebra.cholesky!(pm::PackedMatrix{R}, ::NoPivot = NoPivot(); shift::R = zero(R), check::Bool = true) where {R<:Real}
    if !iszero(shift)
        i = 0
        @inbounds @simd for j in 1:pm.dim
            i += j
            pm.data[i] += shift
        end
    end
    C, info = pptrf!('U', pm.dim, pm.data)
    check && checkpositivedefinite(info)
    return Cholesky(PackedMatrix(pm.dim, C), 'U', info)
end
LinearAlgebra.isposdef(M::PackedMatrix{R}, tol::R=zero(R)) where {R<:Real} =
    isposdef(cholesky!(copy(M), shift=tol, check=false))