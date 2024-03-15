#ifndef CNN_KRNL_H_
#define CNN_KRNL_H_

#define kNum            (256)
#define kKernel         (5)
#define kImSize         (224)
#define kInImSize       (228)
#define kOutImSize      (112)
#define max(X,Y) ((X)>(Y)?(X):(Y))

// template <class T>
// inline T max(T a, T b) { return a > b ? a : b; }

#ifdef FASTSIM

// These code are soly for accelerating software emulation
// and are not used for hardware generation.

// Used for faster software simulation
#pragma GCC target ("arch=skylake")
#pragma GCC optimize ("-O3,-ffast-math")

#define Input(x,y,z)    \
    (input_g[(x)*kInImSize*kInImSize+(y)*kInImSize+(z)])
#define Weight(x,y,z,i) \
    (weight_g[(x)*kNum*kKernel*kKernel+(y)*kKernel*kKernel+(z)*kKernel+(i)])
#define Bias(x)         \
    (bias_g[(x)])
#define Output(x,y,z)   \
    (output_g[(x)*kOutImSize*kOutImSize+(y)*kOutImSize+z])

typedef float input_t;
typedef float weight_t;
typedef float bias_t;
typedef float output_t;
typedef float compute_t;

#else

// #include "ap_fixed.h"
// #include "ap_int.h"

// These code are the actual code used for for hardware generation.

typedef float input_t;
typedef float weight_t;
typedef float bias_t;
typedef float compute_t;
typedef float output_t;

#endif

#endif
