// kernels.hpp — The dense numerics shared by the learners: arena, matmuls, outer products, Adam.
//
// Part of PQL; include "pql/pql.hpp", which brings in every part in order.
#pragma once

#include "pql/common.hpp"

namespace pql {

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------
//
// Every buffer a training run touches is carved from one block sized up front.
// The working set is a known function of (rows, width, hidden), so there is no
// reason to return to the allocator once the shape is known; the hot loops then
// run against memory that is contiguous and already resident.
class Arena {
public:
	void Reserve(size_t bytes) {
		if (buf_.size() < bytes) {
			buf_.resize(bytes);
		}
		used_ = 0;
	}
	// Size the block from the code that carves it. `carve(arena)` runs once
	// against a counting arena, which hands out no memory and only tallies what
	// it would have, and then once for real against the block that tally sized.
	// One code path for both, so the total cannot drift from the allocations:
	// a hand-written total once left the gradient buffers out, and the growth
	// that followed dangled every pointer already handed out.
	template <typename F>
	static size_t Measure(F &&carve) {
		Arena a;
		a.counting_ = true;
		carve(a);
		return a.used_;
	}
	template <typename F>
	void Carve(F &&carve) {
		Reserve(Measure(carve));
		carve(*this);
	}
	template <typename T>
	T *Alloc(size_t n) {
		const size_t align = alignof(T) > 16 ? alignof(T) : 16;
		// n * sizeof(T) wrapping would pass the capacity check and hand out a
		// pointer to far less than was asked for.
		if (n > (std::numeric_limits<size_t>::max() / 2) / sizeof(T)) {
			throw std::runtime_error("pql: arena allocation overflows size_t");
		}
		used_ = (used_ + align - 1) & ~(align - 1);
		const size_t bytes = n * sizeof(T);
		if (counting_) {
			used_ += bytes;
			return nullptr;
		}
		// The alignment step alone can carry used_ past the end of a full block,
		// so both halves are checked; the subtraction never wraps.
		if (used_ > buf_.size() || bytes > buf_.size() - used_) {
			// Growing would reallocate and dangle every pointer already returned.
			throw std::runtime_error("pql: arena exhausted; Reserve() was undersized");
		}
		T *p = reinterpret_cast<T *>(buf_.data() + used_);
		used_ += bytes;
		return p;
	}
	template <typename T>
	T *Zeroed(size_t n) {
		T *p = Alloc<T>(n);
		if (p != nullptr && n > 0) {
			std::memset(p, 0, n * sizeof(T));
		}
		return p;
	}
	void Reset() {
		used_ = 0;
	}
	size_t Used() const {
		return used_;
	}
	size_t Capacity() const {
		return buf_.size();
	}
	bool Counting() const {
		return counting_;
	}

private:
	std::vector<uint8_t> buf_;
	size_t used_ = 0;
	bool counting_ = false;
};

// ---------------------------------------------------------------------------
// Batched kernels. Each takes contiguous float buffers and puts the OUTPUT on
// the innermost, unit-stride axis, with the reduction on an outer loop. That is
// the opposite of the obvious arrangement, where the innermost loop is the dot
// product itself, and it is worth roughly ten times: a dot product per output
// element ends in a horizontal reduction and reloads the weight row for every
// row of the batch, while this shape keeps a block of the output in vector
// registers and touches each operand once per block.
//
// Working a minibatch at a time turns B matrix-vector products into one
// matrix-matrix product and is worth several times the scalar form again.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// One tiled kernel
// ---------------------------------------------------------------------------
//
// A network is a program, and its shape is fixed the moment the model is built.
// QFP makes a program a type so the whole dataflow specialises at compile time,
// and the same idea applies here without a JIT. But the constant worth freezing
// turns out not to be the layer width: it is the register tile. Each kernel
// below reduces along one axis while holding a 4x16 block of the output, and
// because that block is a compile-time shape the compiler keeps all 64
// accumulators in vector registers and unrolls the tile away entirely.
//
// Measured against the previous per-width specialisations that is 3x to 12x
// faster at every shape these models use, including the ones the old table had
// an exact instantiation for, so the table is gone. Writing the tile by hand in
// NEON intrinsics was measured too and matched it to within noise, which is why
// there are none here.
//
// The tile needs the reduction axis to be the *row* axis of the right operand,
// so that neighbours in the output are neighbours in memory. Two of the three
// kernels get that for free. MatMulNT transposes its weights to get it, which
// costs K*N writes against B*K*N multiply-adds.

// C[B x N] = A[B x M] * W[M x N], + bias, + ReLU. With ACC the product is added
// to what C already holds, which is what lets a sum of several linear terms be
// formed in place instead of through a scratch buffer and a second pass.
template <bool BIAS, bool RELU, bool ACC = false>
inline void MatMulNN_Tiled(const float *A, const float *W, const float *bias, float *C, int B, int M, int N) {
	int b = 0;
	for (; b + 4 <= B; b += 4) {
		int n = 0;
		for (; n + 16 <= N; n += 16) {
			float acc[4][16];
			for (int x = 0; x < 4; x++) {
				for (int y = 0; y < 16; y++) {
					acc[x][y] = (ACC ? C[(size_t)(b + x) * (size_t)N + n + y] : 0.0f) + (BIAS ? bias[n + y] : 0.0f);
				}
			}
			for (int m = 0; m < M; m++) {
				const float *wr = W + (size_t)m * (size_t)N + n;
				float a[4];
				for (int x = 0; x < 4; x++) {
					a[x] = A[(size_t)(b + x) * (size_t)M + m];
				}
				for (int x = 0; x < 4; x++) {
					for (int y = 0; y < 16; y++) {
						acc[x][y] += a[x] * wr[y];
					}
				}
			}
			for (int x = 0; x < 4; x++) {
				for (int y = 0; y < 16; y++) {
					const float s = acc[x][y];
					C[(size_t)(b + x) * (size_t)N + n + y] = RELU ? (s > 0.0f ? s : 0.0f) : s;
				}
			}
		}
		for (; n + 4 <= N; n += 4) {
			float acc[4][4];
			for (int x = 0; x < 4; x++) {
				for (int y = 0; y < 4; y++) {
					acc[x][y] = (ACC ? C[(size_t)(b + x) * (size_t)N + n + y] : 0.0f) + (BIAS ? bias[n + y] : 0.0f);
				}
			}
			for (int m = 0; m < M; m++) {
				const float *wr = W + (size_t)m * (size_t)N + n;
				float a[4];
				for (int x = 0; x < 4; x++) {
					a[x] = A[(size_t)(b + x) * (size_t)M + m];
				}
				for (int x = 0; x < 4; x++) {
					for (int y = 0; y < 4; y++) {
						acc[x][y] += a[x] * wr[y];
					}
				}
			}
			for (int x = 0; x < 4; x++) {
				for (int y = 0; y < 4; y++) {
					const float s = acc[x][y];
					C[(size_t)(b + x) * (size_t)N + n + y] = RELU ? (s > 0.0f ? s : 0.0f) : s;
				}
			}
		}
		for (; n < N; n++) {
			for (int x = 0; x < 4; x++) {
				const size_t o = (size_t)(b + x) * (size_t)N + n;
				float s = (ACC ? C[o] : 0.0f) + (BIAS ? bias[n] : 0.0f);
				for (int m = 0; m < M; m++) {
					s += A[(size_t)(b + x) * (size_t)M + m] * W[(size_t)m * (size_t)N + n];
				}
				C[o] = RELU ? (s > 0.0f ? s : 0.0f) : s;
			}
		}
	}
	for (; b < B; b++) {
		for (int n = 0; n < N; n++) {
			const size_t o = (size_t)b * (size_t)N + n;
			float s = (ACC ? C[o] : 0.0f) + (BIAS ? bias[n] : 0.0f);
			for (int m = 0; m < M; m++) {
				s += A[(size_t)b * (size_t)M + m] * W[(size_t)m * (size_t)N + n];
			}
			C[o] = RELU ? (s > 0.0f ? s : 0.0f) : s;
		}
	}
}

// bias and ReLU are template parameters rather than branches so the tile body
// stays free of both a load and a compare.
inline void MatMulNN(const float *A, const float *W, const float *bias, float *C, int B, int M, int N, bool relu) {
	if (B <= 0 || N <= 0) {
		return;
	}
	if (bias) {
		relu ? MatMulNN_Tiled<true, true>(A, W, bias, C, B, M, N) : MatMulNN_Tiled<true, false>(A, W, bias, C, B, M, N);
	} else {
		relu ? MatMulNN_Tiled<false, true>(A, W, bias, C, B, M, N)
		     : MatMulNN_Tiled<false, false>(A, W, bias, C, B, M, N);
	}
}

// Scratch for the weight transpose. Grown on first use and reused for the rest
// of the process, so a training step still reaches the allocator zero times.
inline float *TransposeScratch(size_t n) {
	static thread_local std::vector<float> buf;
	if (buf.size() < n) {
		buf.assign(n, 0.0f);
	}
	return buf.data();
}

// [N x K] -> [K x N], blocked so the strided side of the copy stays within a
// few cache lines instead of striding the whole matrix per element.
inline void TransposeInto(const float *W, float *wt, int K, int N) {
	const int BS = 16;
	for (int j0 = 0; j0 < N; j0 += BS) {
		const int j1 = std::min(j0 + BS, N);
		for (int k0 = 0; k0 < K; k0 += BS) {
			const int k1 = std::min(k0 + BS, K);
			for (int j = j0; j < j1; j++) {
				for (int k = k0; k < k1; k++) {
					wt[(size_t)k * (size_t)N + j] = W[(size_t)j * (size_t)K + k];
				}
			}
		}
	}
}

// C[B x N] = A[B x K] * W[N x K]^T + bias, optionally ReLU'd. W is stored one
// row per output channel, which is the wrong way round for the tile, so it is
// transposed first.
inline void MatMulNT(const float *A, const float *W, const float *bias, float *C, int B, int K, int N, bool relu) {
	if (B <= 0 || N <= 0 || K <= 0) {
		return;
	}
	float *wt = TransposeScratch((size_t)K * (size_t)N);
	TransposeInto(W, wt, K, N);
	MatMulNN(A, wt, bias, C, B, K, N, relu);
}

// C[B x N] += A[B x K] * W[N x K]^T. Several linear terms are summed into one
// pre-activation in both SAGE layers; done this way the sum costs nothing beyond
// the products themselves.
// bias and ReLU belong to the last term of the sum, so the activation can be
// finished inside the same kernel rather than in a pass of its own.
inline void MatMulAccNT(const float *A, const float *W, const float *bias, float *C, int B, int K, int N, bool relu) {
	if (B <= 0 || N <= 0 || K <= 0) {
		return;
	}
	float *wt = TransposeScratch((size_t)K * (size_t)N);
	TransposeInto(W, wt, K, N);
	if (bias) {
		relu ? MatMulNN_Tiled<true, true, true>(A, wt, bias, C, B, K, N)
		     : MatMulNN_Tiled<true, false, true>(A, wt, bias, C, B, K, N);
	} else {
		relu ? MatMulNN_Tiled<false, true, true>(A, wt, nullptr, C, B, K, N)
		     : MatMulNN_Tiled<false, false, true>(A, wt, nullptr, C, B, K, N);
	}
}

// G[N x K] += D[B x N]^T * A[B x K]. The reduction is over B this time, and A's
// rows already run along K, so the tile applies with no transpose.
//
// This is the form for a K worth stepping through sixteen at a time. A relation
// width of three or four reaches neither the 16-wide nor the 4-wide step and
// falls entirely to the scalar tail, so AccumOuter below routes those to a
// compile-time specialisation instead.
//
// Microbenchmarks of these two mispredicted the in-situ result more than once,
// so the split below is by what was measured end to end. A runtime test on K
// inside AccumOuter, rather than a switch the compiler turns into a direct call,
// cost 30% on the dense path even at shapes where the specialised form wins;
// that is why the choice is a switch on exact values and why K above eight, where
// the tile would need more accumulators than there are vector registers, comes
// here instead.
//
// The old kernel skipped a whole row when its D value was exactly zero, which
// ReLU makes true about half the time. Neither tile can branch per element, and
// both are faster anyway.
inline void AccumOuterWideK(const float *D, const float *A, float *G, int B, int K, int N) {
	int j = 0;
	for (; j + 4 <= N; j += 4) {
		int k = 0;
		for (; k + 16 <= K; k += 16) {
			float acc[4][16];
			for (int x = 0; x < 4; x++) {
				for (int y = 0; y < 16; y++) {
					acc[x][y] = 0.0f;
				}
			}
			for (int b = 0; b < B; b++) {
				const float *ar = A + (size_t)b * (size_t)K + k;
				const float *dr = D + (size_t)b * (size_t)N + j;
				for (int x = 0; x < 4; x++) {
					const float d = dr[x];
					for (int y = 0; y < 16; y++) {
						acc[x][y] += d * ar[y];
					}
				}
			}
			for (int x = 0; x < 4; x++) {
				for (int y = 0; y < 16; y++) {
					G[(size_t)(j + x) * (size_t)K + k + y] += acc[x][y];
				}
			}
		}
		for (; k + 4 <= K; k += 4) {
			float acc[4][4];
			for (int x = 0; x < 4; x++) {
				for (int y = 0; y < 4; y++) {
					acc[x][y] = 0.0f;
				}
			}
			for (int b = 0; b < B; b++) {
				const float *ar = A + (size_t)b * (size_t)K + k;
				const float *dr = D + (size_t)b * (size_t)N + j;
				for (int x = 0; x < 4; x++) {
					const float d = dr[x];
					for (int y = 0; y < 4; y++) {
						acc[x][y] += d * ar[y];
					}
				}
			}
			for (int x = 0; x < 4; x++) {
				for (int y = 0; y < 4; y++) {
					G[(size_t)(j + x) * (size_t)K + k + y] += acc[x][y];
				}
			}
		}
		for (; k < K; k++) {
			float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
			for (int b = 0; b < B; b++) {
				const float a = A[(size_t)b * (size_t)K + k];
				const float *dr = D + (size_t)b * (size_t)N + j;
				for (int x = 0; x < 4; x++) {
					acc[x] += dr[x] * a;
				}
			}
			for (int x = 0; x < 4; x++) {
				G[(size_t)(j + x) * (size_t)K + k] += acc[x];
			}
		}
	}
	for (; j < N; j++) {
		for (int k = 0; k < K; k++) {
			float s = 0.0f;
			for (int b = 0; b < B; b++) {
				s += D[(size_t)b * (size_t)N + j] * A[(size_t)b * (size_t)K + k];
			}
			G[(size_t)j * (size_t)K + k] += s;
		}
	}
}

// A relation width is small and known: three or four columns is typical. Both
// the 16-wide and the 4-wide steps along K are then skipped and every column
// goes through the scalar tail, four multiply-adds per five loads. With K a
// compile-time constant the output can be blocked wide instead and the whole K
// dimension held in registers.
//
// Which axis of the accumulator is contiguous decides this kernel. Laid out as
// acc[rows][K] the innermost loop is K long, and K here is three or five: it
// never fills a vector. Transposed to acc[K][rows] the innermost loop walks the
// output block instead, which is a broadcast of one A value against a
// contiguous run of D. At K = 4 that is 10 GFLOP/s against 76, from nothing but
// the swap.
//
// The block is 32 rows up to K = 3 and 16 above, because acc[K][rows] costs
// K*rows/4 vector registers and there are 32; past K = 7 even 16 rows no longer
// fits, and AccumOuterTinyK below (the untransposed form) takes over.
template <int KK, int NW>
inline void AccumOuterTinyStage(const float *D, const float *A, float *G, int B, int N, int &j) {
	for (; j + NW <= N; j += NW) {
		float acc[KK][NW];
		for (int y = 0; y < KK; y++) {
			for (int x = 0; x < NW; x++) {
				acc[y][x] = 0.0f;
			}
		}
		for (int b = 0; b < B; b++) {
			const float *ar = A + (size_t)b * (size_t)KK;
			const float *dr = D + (size_t)b * (size_t)N + j;
			for (int y = 0; y < KK; y++) {
				const float a = ar[y];
				for (int x = 0; x < NW; x++) {
					acc[y][x] += a * dr[x];
				}
			}
		}
		for (int x = 0; x < NW; x++) {
			for (int y = 0; y < KK; y++) {
				G[(size_t)(j + x) * (size_t)KK + y] += acc[y][x];
			}
		}
	}
}

template <int KK, bool WIDE>
inline void AccumOuterTinyT(const float *D, const float *A, float *G, int B, int N) {
	int j = 0;
	if (WIDE) {
		AccumOuterTinyStage<KK, 32>(D, A, G, B, N, j);
	}
	AccumOuterTinyStage<KK, 16>(D, A, G, B, N, j);
	AccumOuterTinyStage<KK, 4>(D, A, G, B, N, j);
	for (; j < N; j++) {
		for (int k = 0; k < KK; k++) {
			float s = 0.0f;
			for (int b = 0; b < B; b++) {
				s += D[(size_t)b * (size_t)N + j] * A[(size_t)b * (size_t)KK + k];
			}
			G[(size_t)j * (size_t)KK + k] += s;
		}
	}
}

// The untransposed form, which wins once K fills the registers on its own.
template <int KK>
inline void AccumOuterTinyK(const float *D, const float *A, float *G, int B, int N) {
	int j = 0;
	for (; j + 16 <= N; j += 16) {
		float acc[16][KK];
		for (int x = 0; x < 16; x++) {
			for (int y = 0; y < KK; y++) {
				acc[x][y] = 0.0f;
			}
		}
		for (int b = 0; b < B; b++) {
			const float *ar = A + (size_t)b * (size_t)KK;
			const float *dr = D + (size_t)b * (size_t)N + j;
			for (int x = 0; x < 16; x++) {
				const float d = dr[x];
				for (int y = 0; y < KK; y++) {
					acc[x][y] += d * ar[y];
				}
			}
		}
		for (int x = 0; x < 16; x++) {
			for (int y = 0; y < KK; y++) {
				G[(size_t)(j + x) * (size_t)KK + y] += acc[x][y];
			}
		}
	}
	for (; j < N; j++) {
		for (int k = 0; k < KK; k++) {
			float s = 0.0f;
			for (int b = 0; b < B; b++) {
				s += D[(size_t)b * (size_t)N + j] * A[(size_t)b * (size_t)KK + k];
			}
			G[(size_t)j * (size_t)KK + k] += s;
		}
	}
}

inline void AccumOuter(const float *D, const float *A, float *G, int B, int K, int N) {
	// The boundaries are where the measurements crossed, not where they look
	// tidy: 32-wide blocks up to K = 3, 16-wide through K = 7, then the
	// untransposed tile at K = 8, then the general kernel.
	switch (K) {
	case 1:
		AccumOuterTinyT<1, true>(D, A, G, B, N);
		return;
	case 2:
		AccumOuterTinyT<2, true>(D, A, G, B, N);
		return;
	case 3:
		AccumOuterTinyT<3, true>(D, A, G, B, N);
		return;
	case 4:
		AccumOuterTinyT<4, false>(D, A, G, B, N);
		return;
	case 5:
		AccumOuterTinyT<5, false>(D, A, G, B, N);
		return;
	case 6:
		AccumOuterTinyT<6, false>(D, A, G, B, N);
		return;
	case 7:
		AccumOuterTinyT<7, false>(D, A, G, B, N);
		return;
	case 8:
		AccumOuterTinyK<8>(D, A, G, B, N);
		return;
	default:
		AccumOuterWideK(D, A, G, B, K, N);
		return;
	}
}

// dA[B x K] = D[B x N] * W[N x K], masked by the pre-activation sign of H.
// W is stored [N x K], so K is already the contiguous axis and this is the
// plain tiled product with no transpose at all.
inline void BackThroughRelu(const float *D, const float *W, const float *H, float *dA, int B, int K, int N) {
	MatMulNN(D, W, nullptr, dA, B, N, K, false);
	for (int b = 0; b < B; b++) {
		float *arow = dA + (size_t)b * (size_t)K;
		const float *hrow = H + (size_t)b * (size_t)K;
		// Mask by multiply rather than branch: the ReLU gate is data dependent and
		// splits roughly evenly, which is the worst case for a predictor.
		for (int k = 0; k < K; k++) {
			arow[k] *= (hrow[k] > 0.0f) ? 1.0f : 0.0f;
		}
	}
}

// out[b] = bias + dot(w, H[b]). Written as four rows by four lanes rather than
// one accumulator per row: a single running sum makes the reduction a dependent
// chain 'hidden' long, and at these sizes that latency, not the arithmetic, was
// what the loop spent its time on.
inline void OutputLayer(const float *H, const float *w, float bias, float *out, int B, int hidden) {
	const int h4 = (hidden / 4) * 4;
	int b = 0;
	for (; b + 4 <= B; b += 4) {
		float acc[4][4];
		for (int x = 0; x < 4; x++) {
			for (int y = 0; y < 4; y++) {
				acc[x][y] = 0.0f;
			}
		}
		for (int j = 0; j < h4; j += 4) {
			for (int x = 0; x < 4; x++) {
				const float *hr = H + (size_t)(b + x) * (size_t)hidden + j;
				for (int y = 0; y < 4; y++) {
					acc[x][y] += w[j + y] * hr[y];
				}
			}
		}
		for (int x = 0; x < 4; x++) {
			float s = ((acc[x][0] + acc[x][1]) + (acc[x][2] + acc[x][3])) + bias;
			const float *hr = H + (size_t)(b + x) * (size_t)hidden;
			for (int j = h4; j < hidden; j++) {
				s += w[j] * hr[j];
			}
			out[b + x] = s;
		}
	}
	for (; b < B; b++) {
		const float *hr = H + (size_t)b * (size_t)hidden;
		float a0 = bias, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
		int j = 0;
		for (; j + 4 <= hidden; j += 4) {
			a0 += w[j] * hr[j];
			a1 += w[j + 1] * hr[j + 1];
			a2 += w[j + 2] * hr[j + 2];
			a3 += w[j + 3] * hr[j + 3];
		}
		float s = (a0 + a1) + (a2 + a3);
		for (; j < hidden; j++) {
			s += w[j] * hr[j];
		}
		out[b] = s;
	}
}

// Score a batch through the same kernels training uses. Scratch is supplied by
// the caller so the evaluation path allocates nothing per epoch.
struct EvalScratch {
	// Scoring is not bound to the training batch size. Its own buffer lets a whole
	// fold go through in a few passes rather than one per 64 rows, which amortises
	// the weight transpose each pass pays for.
	static const size_t ROWS = 512;
	std::vector<float> h1, h2, o, x;
	void Ensure(size_t rows, int hidden, int width) {
		if (h1.size() < rows * (size_t)hidden) {
			h1.assign(rows * (size_t)hidden, 0.0f);
			h2.assign(rows * (size_t)hidden, 0.0f);
		}
		if (o.size() < rows) {
			o.assign(rows, 0.0f);
		}
		if (x.size() < rows * (size_t)std::max(1, width)) {
			x.assign(rows * (size_t)std::max(1, width), 0.0f);
		}
	}
};

// Adam over a flat parameter block; one instance per tensor.
//
// This used to do its arithmetic in double over float arrays, which forced a
// widen and a narrow per element and halved the usable vector width. The state
// is stored as float either way, so the extra precision never survived past the
// next store. Measured drift over a full 7120-step run is 1.7e-4 relative, and
// the update is four to six times faster.
//
// Both bias corrections are folded into scalars ahead of the loop, which leaves
// exactly one divide and one square root per parameter: the ones Adam needs.
struct Adam {
	std::vector<float> m, v;
	int t = 0;
	// `wd` is decoupled weight decay: it pulls weights toward zero directly
	// rather than through the gradient, so the adaptive step size does not scale
	// it away. Biases are stepped with wd = 0; shrinking an intercept just biases
	// the model without constraining it.
	void Step(std::vector<float> &w, const float *g, double lr, double wd = 0.0) {
		if (m.size() != w.size()) {
			m.assign(w.size(), 0.0f);
			v.assign(w.size(), 0.0f);
		}
		t++;
		const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
		const double c1 = 1.0 - std::pow(b1, t), c2 = 1.0 - std::pow(b2, t);
		const float f1 = (float)b1, f2 = (float)b2;
		const float o1 = (float)(1.0 - b1), o2 = (float)(1.0 - b2);
		const float alpha = (float)(lr / c1), rc2 = (float)(1.0 / std::sqrt(c2));
		const float fe = (float)eps;
		const float decay = (float)(lr * wd);
		float *__restrict mp = m.data();
		float *__restrict vp = v.data();
		float *__restrict wp = w.data();
		const size_t n = w.size();
		if (decay == 0.0f) {
			for (size_t i = 0; i < n; i++) {
				const float gi = g[i];
				const float mi = f1 * mp[i] + o1 * gi;
				const float vi = f2 * vp[i] + o2 * gi * gi;
				mp[i] = mi;
				vp[i] = vi;
				wp[i] -= alpha * mi / (std::sqrt(vi) * rc2 + fe);
			}
			return;
		}
		for (size_t i = 0; i < n; i++) {
			const float gi = g[i];
			const float mi = f1 * mp[i] + o1 * gi;
			const float vi = f2 * vp[i] + o2 * gi * gi;
			mp[i] = mi;
			vp[i] = vi;
			wp[i] -= alpha * mi / (std::sqrt(vi) * rc2 + fe) + decay * wp[i];
		}
	}
};

} // namespace pql
