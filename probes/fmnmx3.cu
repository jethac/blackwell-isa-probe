// Gap-2 (footnote 8): 3-input min/max SASS fusion probe.
// Chained min/max, both float and int, compiled at -O3. cuobjdump --dump-sass,
// then grep for FMNMX3 / VIMNMX3 / IMNMX3 (3-input, fused in HW) vs pairs of
// FMNMX / IMNMX (2-input, no HW 3-input unit).
// Loads from global + stores to distinct slots to defeat DCE / const-fold.
extern "C" __global__ void f3(const float* __restrict__ in, float* __restrict__ out){
    float a=in[0], b=in[1], c=in[2], d=in[3], e=in[4];
    out[0] = fmaxf(fmaxf(a,b),c);                      // float 3-max
    out[1] = fminf(fminf(a,b),c);                      // float 3-min
    out[2] = fmaxf(fmaxf(fmaxf(fmaxf(a,b),c),d),e);    // float 5-max (amplify: fusion -> fewer instrs)
    out[3] = fminf(fminf(fminf(fminf(a,b),c),d),e);    // float 5-min
}
extern "C" __global__ void i3(const int* __restrict__ in, int* __restrict__ out){
    int a=in[0], b=in[1], c=in[2], d=in[3], e=in[4];
    out[0] = max(max(a,b),c);                          // int 3-max (signed)
    out[1] = min(min(a,b),c);                          // int 3-min (signed)
    out[2] = max(max(max(max(a,b),c),d),e);            // int 5-max
    out[3] = min(min(min(min(a,b),c),d),e);            // int 5-min
}
extern "C" __global__ void u3(const unsigned* __restrict__ in, unsigned* __restrict__ out){
    unsigned a=in[0], b=in[1], c=in[2];
    out[0] = max(max(a,b),c);                          // uint 3-max
    out[1] = min(min(a,b),c);                          // uint 3-min
}
