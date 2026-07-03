// j_multicast_tma.cu -- SASS check for ISA-table footnote 3:
// does TMA .multicast::cluster lower to a real UTMALDG.2D.MULTICAST on sm120,
// or (as the footnote claims for 12.x/13.x) to a syscall + a plain unicast load?
// Compile + cuobjdump only:  nvcc -arch=sm_120a -cubin -o mc.cubin j_multicast_tma.cu
#include <cuda_runtime.h>
#include <cstdint>

// multicast::cluster form -- one TMA load broadcast to the shared memory of every
// CTA in the cluster named by ctaMask.
__global__ void tma_multicast(const void* __restrict__ tmap, uint32_t cta_mask) {
  __shared__ __align__(128) uint8_t  buf[1024];
  __shared__ __align__(8)   uint64_t bar;
  uint32_t dst  = static_cast<uint32_t>(__cvta_generic_to_shared(buf));
  uint32_t mbar = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster "
    "[%0], [%1, {%2, %3}], [%4], %5;"
    :: "r"(dst), "l"(tmap), "r"(0), "r"(0), "r"(mbar), "h"((uint16_t)cta_mask)
    : "memory");
}

// plain (unicast) form -- baseline for the SASS diff.
__global__ void tma_unicast(const void* __restrict__ tmap) {
  __shared__ __align__(128) uint8_t  buf[1024];
  __shared__ __align__(8)   uint64_t bar;
  uint32_t dst  = static_cast<uint32_t>(__cvta_generic_to_shared(buf));
  uint32_t mbar = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
    "[%0], [%1, {%2, %3}], [%4];"
    :: "r"(dst), "l"(tmap), "r"(0), "r"(0), "r"(mbar)
    : "memory");
}
