#pragma once

#include "config.cuh"


#define C_FLT_MAX INFINITY


////////////////////////////////////////////////////////////////////////////////
// TODO replace: Placeholders

template <typename T>
inline __device__ T Min(T lhs, T rhs) {	return lhs < rhs ? lhs : rhs; }

template <typename T>
inline __device__ T Max(T lhs, T rhs) {	return lhs < rhs ? rhs : lhs; }

template <typename T>
inline __device__ T Clamp(T x, T a, T b) { return Max(Min(x, b), a); }

template <typename T>
inline __device__ T Saturate(T x) { return Clamp(x, T(0), T(1)); }

template <typename T>
inline __device__ T Sqrt(T x) { return sqrt(x); }



template <typename T>
struct tcvec2
{
	T x, y;

	static __device__ tcvec2 c(T x, T y) { return { x, y }; }

	__device__ tcvec2 operator+(tcvec2 const & o) const { return tcvec2::c(x + o.x, y + o.y); }
	__device__ tcvec2 operator-(tcvec2 const & o) const { return tcvec2::c(x - o.x, y - o.y); }
	__device__ tcvec2 operator*(tcvec2 const & o) const { return tcvec2::c(x * o.x, y * o.y); }
	__device__ tcvec2 operator/(tcvec2 const & o) const { return tcvec2::c(x / o.x, y / o.y); }

	__device__ tcvec2 operator+() const { return tcvec2::c(+x,+ y); }
	__device__ tcvec2 operator-() const { return tcvec2::c(-x, -y); }

	__device__ tcvec2 const & operator-=(tcvec2<T> const & v) { x -= v.x; y -= v.y; return *this; }

	__device__ tcvec2 const & operator+=(T v) { x += v; y += v; return *this; }
	__device__ tcvec2 const & operator*=(T v) { x *= v; y *= v; return *this; }
	__device__ tcvec2 const & operator/=(T v) { x /= v; y /= v; return *this; }
};

template <typename T>
struct tvec2 : public tcvec2<T>
{
	__device__ explicit tvec2(T v = T(0)) { x = y = v; }
	__device__ tvec2(T xx, T yy) { x = xx; y = yy; }
	template <typename U>
	__device__ tvec2(tvec2<U> const & o) { x = o.x; y = o.y; }
	template <typename U>
	__device__ tvec2(tcvec2<U> const & o) { x = o.x; y = o.y; }
	__device__ tvec2(float2 const & o) { x = o.x; y = o.y; }
};

template <typename T>
inline __device__ tcvec2<T> operator+(tcvec2<T> const & l, tcvec2<T> const & r) { return tcvec2<T>::c(l.x + r.x, l.y + r.y); }

template <typename T>
inline __device__ tcvec2<T> operator-(tcvec2<T> const & l, tcvec2<T> const & r) { return tcvec2<T>::c(l.x - r.x, l.y - r.y); }

template <typename T>
inline __device__ tcvec2<T> operator-(T x, tcvec2<T> const & v) {	return tcvec2<T>::c(x - v.x, x - v.y); }

template <typename T>
inline __device__ tcvec2<T> operator-(tcvec2<T> const & v, T x) {	return tcvec2<T>::c(v.x - x, v.y - x); }

template <typename T>
inline __device__ tcvec2<T> operator+(tcvec2<T> const & v, T x) {	return tcvec2<T>::c(v.x + x, v.y + x); }


using icvec2 = tcvec2<int>;
using ivec2  = tvec2<int>;
using vec2   = tvec2<float>;


template <typename T>
struct tcvec3
{
	T x, y, z;
};

using cvec3 = tcvec3<float>;

template <typename T>
struct tvec3 : public tcvec3<T>
{
	__device__ explicit tvec3(T v = T(0)) { x = v; y = v; z = v; }
	__device__ tvec3(T xx, T yy, T zz) { x = xx; y = yy; z = zz; }
	__device__ tvec3(tcvec3<T> const & o) { x = o.x; y = o.y; z = o.z; }
	__device__ tvec3 & operator=(tvec3 const & o) { x = o.x; y = o.y; z = o.z; return *this; }
	__device__ tvec3 operator+(tvec3 const & o) const { return tvec3(x + o.x, y + o.y, z + o.z); }
	__device__ tvec3 operator-(tvec3 const & o) const { return tvec3(x - o.x, y - o.y, z - o.z); }
	__device__ tvec3 operator*(tvec3 const & o) const { return tvec3(x * o.x, y * o.y, z * o.z); }

	__device__ tvec3 const & operator+=(tvec3 const & o) { x += o.x; y += o.y; z += o.z; return *this; }

	__device__ tvec3 const & operator/=(T v) { x /= v; y /= v; z /= v; return *this; }

	__device__ tvec3 operator/(tvec3<T> const & v) { return tvec3<T>(x / v.x, y / v.y, z / v.z); }
};

template <typename T>
inline __device__ tvec3<T> operator*(T x, tvec3<T> const & v) {	return tvec3<T>(x * v.x, x * v.y, x * v.z); }

template <typename T>
inline __device__ tvec3<T> operator*(tvec3<T> const & v, T x) {	return tvec3<T>(x * v.x, x * v.y, x * v.z); }

template <typename T>
inline __device__ tvec3<T> operator/(tvec3<T> const & v, T x) {	return tvec3<T>(v.x / x, v.y / x, v.z / x); }


template <typename T>
inline __device__ tvec3<T> Min(tvec3<T> const & l, tvec3<T> const & r)
{
	return tvec3<T>(Min(l.x, r.x), Min(l.y, r.y), Min(l.z, r.z));
}

template <typename T>
inline __device__ tvec3<T> Max(tvec3<T> const & l, tvec3<T> const & r)
{
	return tvec3<T>(Max(l.x, r.x), Max(l.y, r.y), Max(l.z, r.z));
}

template <typename T>
inline __device__ tvec3<T> Clamp(tvec3<T> const & v, tvec3<T> const & a, tvec3<T> const & b)
{
	return tvec3<T>(Clamp(v.x, a.x, b.x), Clamp(v.y, a.y, b.y), Clamp(v.z, a.z, b.z));
}

template <typename T>
inline __device__ tvec3<T> Pow(tvec3<T> const & v, T p)
{
	return tvec3<T>(pow(v.x, p), pow(v.y, p), pow(v.z, p));
}

template <typename T>
inline __device__ T Lerp(T a, T b, T t)
{
	return (T(1) - t) * a + t * b;
}

template <typename T>
inline __device__ tvec3<T> Lerp(tvec3<T> const & a, tvec3<T> const & b, tvec3<T> const & t)
{
	return tvec3<T>(Lerp(a.x, b.x, t.x), Lerp(a.y, b.y, t.y), Lerp(a.z, b.z, t.z));
}

template <typename T>
inline __device__ tvec3<T> Lerp(tvec3<T> const & a, tvec3<T> const & b, T t)
{
	return tvec3<T>(Lerp(a.x, b.x, t), Lerp(a.y, b.y, t), Lerp(a.z, b.z, t));
}

using ivec3 = tvec3<int>;
using vec3 = tvec3<float>;

template <typename T>
inline __device__ T Dot(tvec3<T> const & lhs, tvec3<T> const & rhs) { return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z; }

template <typename T>
struct tcvec4
{
	T x, y, z, w;
};

using cvec4 = tcvec4<float>;

template <typename T>
struct tvec4 : public tcvec4<T>
{
	__device__ explicit tvec4(T v = T(0)) { x = v; y = v; z = v; w = v; }
	__device__ tvec4(T xx, T yy, T zz, T ww) { x = xx; y = yy; z = zz; w = ww; }
	__device__ tvec4(tvec3<T> const & v, T ww) { x = v.x; y = v.y; z = v.z; w = ww; }
	__device__ tvec4(tcvec4<T> const & o) { x = o.x; y = o.y; z = o.z; w = o.w; }
	__device__ tvec4 operator+(tvec4 const & o) const { return tvec4(x + o.x, y + o.y, z + o.z, w + o.w); }
	__device__ tvec4 operator-(tvec4 const & o) const { return tvec4(x - o.x, y - o.y, z - o.z, w - o.w); }

	__device__ T operator[](int i) const { return *((&x) + i); }

	__device__ tvec3<T> xyz() const { return tvec3<T>(x, y, z); }
};

using ivec4 = tvec4<int>;
using vec4 = tvec4<float>;

template <typename T>
struct tmat4x4
{
	tvec4<T> m[4]; // Column major
	__device__ tvec4<T> row(int i) const { return tvec4<T>(m[0][i], m[1][i], m[2][i], m[3][i]); }
	__device__ tvec4<T> const & operator[](int i) const { return m[i]; }
};

using mat4x4 = tmat4x4<float>;

template <typename T>
inline __device__ T Dot(tvec4<T> const & lhs, tvec4<T> const & rhs) { return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z + lhs.w*rhs.w; }

template <typename T>
inline __device__ T Abs(T x) { return x < 0 ? -x : x; }


#ifdef __CUDACC__

inline __device__ half Add(half lhs, half rhs)
{
	#if K_SUPPORT_HALF16_ARITHMETIC
	return __hadd(lhs, rhs);
	#else
	return __float2half(__half2float(lhs) + __half2float(rhs));
	#endif
}

inline __device__ void Add3(half const * lhs, half const * rhs, half * res)
{
	#if K_SUPPORT_HALF16_ARITHMETIC
	half2 tmp = __hadd2(__halves2half2(lhs[0], lhs[1]), __halves2half2(rhs[0], rhs[1]));
	res[0] = __low2half(tmp);
	res[1] = __high2half(tmp);
	res[2] = __hadd(lhs[2], rhs[2]);
	#else
	res[0] = __float2half(__half2float(lhs[0]) + __half2float(rhs[0]));
	res[1] = __float2half(__half2float(lhs[1]) + __half2float(rhs[1]));
	res[2] = __float2half(__half2float(lhs[2]) + __half2float(rhs[2]));
	#endif
}

inline __device__ void Add(half lhs[3], half rhs[3], half res[3])
{
	Add3(lhs, rhs, res);
}

inline __device__ half Sub(half lhs, half rhs)
{
	#if K_SUPPORT_HALF16_ARITHMETIC
	return __hsub(lhs, rhs);
	#else
	return __float2half(__half2float(lhs) - __half2float(rhs));
	#endif
}

inline __device__ void Sub3(half const * lhs, half const * rhs, half * res)
{
	#if K_SUPPORT_HALF16_ARITHMETIC
	half2 tmp = __hsub2(__halves2half2(lhs[0], lhs[1]), __halves2half2(rhs[0], rhs[1]));
	res[0] = __low2half(tmp);
	res[1] = __high2half(tmp);
	res[2] = __hsub(lhs[2], rhs[2]);
	#else
	res[0] = __float2half(__half2float(lhs[0]) - __half2float(rhs[0]));
	res[1] = __float2half(__half2float(lhs[1]) - __half2float(rhs[1]));
	res[2] = __float2half(__half2float(lhs[2]) - __half2float(rhs[2]));
	#endif
}

inline __device__ void Sub(half lhs[3], half rhs[3], half res[3])
{
	Sub3(lhs, rhs, res);
}

inline __device__ half Mul(half lhs, half rhs)
{
	#if K_SUPPORT_HALF16_ARITHMETIC
	return __hmul(lhs, rhs);
	#else
	return __float2half(__half2float(lhs) * __half2float(rhs));
	#endif
}

inline __device__ void Mul3(half const * lhs, half const * rhs, half * res)
{
	#if K_SUPPORT_HALF16_ARITHMETIC
	half2 tmp = __hmul2(__halves2half2(lhs[0], lhs[1]), __halves2half2(rhs[0], rhs[1]));
	res[0] = __low2half(tmp);
	res[1] = __high2half(tmp);
	res[2] = __hmul(lhs[2], rhs[2]);
	#else
	res[0] = __float2half(__half2float(lhs[0]) * __half2float(rhs[0]));
	res[1] = __float2half(__half2float(lhs[1]) * __half2float(rhs[1]));
	res[2] = __float2half(__half2float(lhs[2]) * __half2float(rhs[2]));
	#endif
}

inline __device__ void Mul(half lhs[3], half rhs[3], half res[3])
{
	Mul3(lhs, rhs, res);
}

inline __device__ half Div(half lhs, half rhs)
{
	#if K_SUPPORT_HALF16_ARITHMETIC
	return __hdiv(lhs, rhs);
	#else
	return __float2half(__half2float(lhs) / __half2float(rhs));
	#endif
}

inline __device__ void Div3(half const * lhs, half const * rhs, half * res)
{
	#if K_SUPPORT_HALF16_ARITHMETIC
	half2 tmp = __h2div(__halves2half2(lhs[0], lhs[1]), __halves2half2(rhs[0], rhs[1]));
	res[0] = __low2half(tmp);
	res[1] = __high2half(tmp);
	res[2] = __hdiv(lhs[2], rhs[2]);
	#else
	res[0] = __float2half(__half2float(lhs[0]) / __half2float(rhs[0]));
	res[1] = __float2half(__half2float(lhs[1]) / __half2float(rhs[1]));
	res[2] = __float2half(__half2float(lhs[2]) / __half2float(rhs[2]));
	#endif
}

inline __device__ void Div(half lhs[3], half rhs[3], half res[3])
{
	Div3(lhs, rhs, res);
}

inline __device__ half Sqrt(half x)
{
	return __float2half(sqrt(__half2float(x)));
}


template <>
struct tvec3<half> : public tcvec3<half>
{
	__device__ tvec3() { x = { 0 }; y = { 0 }; z = { 0 }; }
	__device__ explicit tvec3(half v) { x = y = z = v; }
	__device__ tvec3(half xx, half yy, half zz) { x = xx; y = yy; z = zz; }
	__device__ tvec3 & operator=(tvec3 const & o) { x = o.x; y = o.y; z = o.z; return *this; }
	__device__ tvec3 operator+(tvec3 const & o) const { tvec3 v; Add3(&x, &o.x, &v.x); return v; }
	__device__ tvec3 operator-(tvec3 const & o) const { tvec3 v; Sub3(&x, &o.x, &v.x); return v; }
	__device__ tvec3 operator*(tvec3 const & o) const { tvec3 v; Mul3(&x, &o.x, &v.x); return v; }
	__device__ tvec3 operator/(tvec3 const & o) const { tvec3 v; Div3(&x, &o.x, &v.x); return v; }

	__device__ tvec3 const & operator+=(tvec3 const & o) { Add3(&x, &o.x, &x); return *this; }
	__device__ tvec3 const & operator-=(tvec3 const & o) { Sub3(&x, &o.x, &x); return *this; }
	__device__ tvec3 const & operator*=(tvec3 const & o) { Mul3(&x, &o.x, &x); return *this; }
	__device__ tvec3 const & operator/=(tvec3 const & o) { Div3(&x, &o.x, &x); return *this; }

	__device__ tvec3 const & operator+=(half v) { half vv[3] = { v, v, v }; Add3(&x, vv, &x); return *this; }
	__device__ tvec3 const & operator-=(half v) { half vv[3] = { v, v, v }; Sub3(&x, vv, &x); return *this; }
	__device__ tvec3 const & operator*=(half v) { half vv[3] = { v, v, v }; Mul3(&x, vv, &x); return *this; }
	__device__ tvec3 const & operator/=(half v) { half vv[3] = { v, v, v }; Div3(&x, vv, &x); return *this; }
};

template <>
struct tvec4<half> : public tcvec4<half>
{
	__device__ tvec3<half> xyz() const { return tvec3<half>(x, y, z); }
};

using vec3h = tvec3<half>;
using vec4h = tvec4<half>;

inline __device__ half Lerp(half a, half b, half t)
{
	#if K_SUPPORT_HALF16_ARITHMETIC
	half one_minus_t = __hsub(__float2half(1.0f), t);
	half2 partial_sums = __hmul2(__halves2half2(one_minus_t, t), __halves2half2(a, b));
	return __hadd(__low2half(partial_sums), __high2half(partial_sums));
	#else
	float ft = __half2float(t);
	float fa = __half2float(a);
	float fb = __half2float(b);
	return __float2half((1.0f - ft) * fa + ft * fb);
	#endif
}

// TODO: benchmark
inline __device__ vec3h Lerp(vec3h const & a, vec3h const & b, vec3h const & t)
{
	#if 0
	half h1 = __float2half(1.0f);

	half2 one_minus_t_xy = __hsub2(__halves2half2(h1, t.x), __halves2half2(h1, t.y));
	half2 lhs_xy = __hmul2(one_minus_t_xy, __halves2half2(a.x, a.y));
	half2 rhs_xy = __hmul2(__halves2half2(t.x, t.y), __halves2half2(b.x, b.y));
	half2 res_xy = __hadd2(lhs_xy, rhs_xy);

	half one_minus_t_z = __hsub(h1, t.z);
	half2 partial_sums_z = __hmul2(__halves2half2(one_minus_t_z, t.z), __halves2half2(a.z, b.z));
	half res_z = __hadd(__low2half(partial_sums_z), __high2half(partial_sums_z));

	return vec3h(__low2half(res_xy), __high2half(res_xy), res_z);
	#else
	return vec3h(Lerp(a.x, b.x, t.x), Lerp(a.y, b.y, t.y), Lerp(a.z, b.z, t.z));
	#endif
}

// TODO: benchmark
inline __device__ vec3h Lerp(vec3h const & a, vec3h const & b, half t)
{
	#if 0
	half one_minus_t = __hsub(__float2half(1.0f), t);

	half2 lhs_xy = __hmul2(__halves2half2(one_minus_t, one_minus_t), __halves2half2(a.x, a.y));
	half2 rhs_xy = __hmul2(__halves2half2(t, t), __halves2half2(b.x, b.y));
	half2 res_xy = __hadd2(lhs_xy, rhs_xy);

	half2 partial_sums_z = __hmul2(__halves2half2(one_minus_t, t), __halves2half2(a.z, b.z));
	half res_z = __hadd(__low2half(partial_sums_z), __high2half(partial_sums_z));

	return vec3h(__low2half(res_xy), __high2half(res_xy), res_z);
	#else
	return vec3h(Lerp(a.x, b.x, t), Lerp(a.y, b.y, t), Lerp(a.z, b.z, t));
	#endif
}

#endif //__CUDACC__
