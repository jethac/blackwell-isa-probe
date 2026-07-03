// j_cluster_probe.cu
// DEFINITIVE runtime test: can consumer Blackwell (sm120 / RTX PRO 6000) actually
// LAUNCH a multi-CTA thread-block cluster (CGA) at runtime, and do distributed
// shared memory (DSM) + cluster.sync actually WORK on silicon -- not just assemble?
// Closes the ISA-table "Clusters" row footnote 4 / "compile oracle can't see the
// runtime" caveat (ISA_TABLE_BLOG_DRAFT.md).
//
// Build:  nvcc -arch=sm_120a -o j_cluster_probe j_cluster_probe.cu
//   (family target also works: nvcc -arch=sm_120f ...)
//
// Answers, with hard runtime evidence:
//   (1) cudaOccupancyMaxPotentialClusterSize / cudaOccupancyMaxActiveClusters
//   (2) actual cudaLaunchKernelEx at clusterDim = 2,4,8,16 -> largest that launches
//   (3) DSM correctness: every rank reads rank 0's shared memory via map_shared_rank
//   (4) cluster.sync(): a producer/consumer chain across ranks that races w/o it

#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  printf("  CUDA ERROR %s:%d: %s -> %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(_e)); } } while(0)

// out[] layout per block: {num_blocks, dsm_read_rank0, chain_read_prev, remote_store}
__global__ void cluster_probe(int* out, int sentinel) {
  cg::cluster_group cluster = cg::this_cluster();
  __shared__ int smem[4];
  unsigned rank = cluster.block_rank();
  unsigned n    = cluster.num_blocks();

  // Producers: ONLY rank 0 stamps the sentinel into ITS shared memory. Every rank
  // publishes its own token in smem[1]; smem[2] is a mailbox a peer will write.
  if (threadIdx.x == 0) {
    smem[0] = (rank == 0) ? sentinel : -1;
    smem[1] = 7000 + (int)rank;
    smem[2] = -1;
  }
  cluster.sync();                     // cluster-wide barrier (barrier.cluster)

  int dsm_read = -777, chain_read = -777;
  if (threadIdx.x == 0) {
    // (3) DSM: EVERY rank maps rank 0's shared window and reads the sentinel.
    dsm_read = cluster.map_shared_rank(smem, 0)[0];             // expect sentinel
    // (4) producer/consumer across ranks, ordered by the barrier above:
    unsigned prev = (rank + n - 1) % n;
    unsigned next = (rank + 1) % n;
    chain_read = cluster.map_shared_rank(smem, prev)[1];        // expect 7000+prev
    cluster.map_shared_rank(smem, next)[2] = 9000 + (int)rank;  // remote store
  }
  cluster.sync();                     // ensure the remote stores are visible
  if (threadIdx.x == 0) {
    int g = blockIdx.x;
    out[g*4+0] = (int)n;
    out[g*4+1] = dsm_read;            // sentinel
    out[g*4+2] = chain_read;          // 7000+prev
    out[g*4+3] = smem[2];             // 9000+prev  (peer's remote store landed here)
  }
}

// Race demonstrator: identical DSM read but NO middle barrier, and rank 0 delays its
// sentinel write ~3ms so peers reading rank 0's smem lose the race and read a stale
// value. A mismatch here proves the barrier in cluster_probe is load-bearing.
__global__ void cluster_probe_race(int* out, int sentinel) {
  cg::cluster_group cluster = cg::this_cluster();
  __shared__ int smem[4];
  unsigned rank = cluster.block_rank();
  if (threadIdx.x == 0) {
    smem[0] = -1;                                       // "not written yet"
    if (rank == 0) { __nanosleep(3000000); smem[0] = sentinel; }  // ~3ms late write
  }
  // NO cluster.sync() here -- deliberate race.
  int dsm_read = -777;
  if (threadIdx.x == 0) dsm_read = cluster.map_shared_rank(smem, 0)[0];
  cluster.sync();
  if (threadIdx.x == 0) out[blockIdx.x] = dsm_read;     // peers likely != sentinel
}

int main() {
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  int drv=0, rt=0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
  printf("device=%s cc=%d.%d SMs=%d driverCUDA=%d.%d runtimeCUDA=%d.%d\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount,
         drv/1000,(drv%1000)/10, rt/1000,(rt%1000)/10);

  int clusterLaunch=-1;
  CK(cudaDeviceGetAttribute(&clusterLaunch, cudaDevAttrClusterLaunch, 0));
  printf("cudaDevAttrClusterLaunch = %d\n", clusterLaunch);

  // (1) Occupancy queries -------------------------------------------------------
  cudaLaunchConfig_t occ = {};
  occ.blockDim = dim3(128,1,1);
  occ.gridDim  = dim3(prop.multiProcessorCount,1,1);
  occ.dynamicSmemBytes = 0;
  int maxPot=-1;
  CK(cudaOccupancyMaxPotentialClusterSize(&maxPot, (void*)cluster_probe, &occ));
  printf("cudaOccupancyMaxPotentialClusterSize = %d\n", maxPot);
  for (int cd : {2,4,8,16}) {
    cudaLaunchAttribute a{};
    a.id = cudaLaunchAttributeClusterDimension;
    a.val.clusterDim = {(unsigned)cd,1,1};
    cudaLaunchConfig_t c = {};
    c.blockDim = dim3(128,1,1);
    c.gridDim  = dim3(prop.multiProcessorCount,1,1);
    c.attrs = &a; c.numAttrs = 1;
    int nc=-1;
    cudaError_t e = cudaOccupancyMaxActiveClusters(&nc, (void*)cluster_probe, &c);
    printf("cudaOccupancyMaxActiveClusters[clusterDim=%d] = %d (%s)\n",
           cd, nc, cudaGetErrorString(e));
  }

  // (2)+(3)+(4) Launch at increasing cluster sizes; verify DSM + barrier ---------
  const int NCL = 2;                             // launch 2 clusters each time
  int* dout; CK(cudaMalloc(&dout, 16*NCL*4*sizeof(int)));
  int hout[16*NCL*4];
  int largest_ok = 0;
  for (int cs : {2,4,8,16}) {
    if (cs > 8) {
      cudaError_t enp = cudaFuncSetAttribute((const void*)cluster_probe,
        cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      printf("[cs=%2d] NonPortableClusterSizeAllowed set: %s\n", cs, cudaGetErrorString(enp));
    }
    int sentinel = 424200 + cs;
    CK(cudaMemset(dout, 0xff, cs*NCL*4*sizeof(int)));
    cudaLaunchConfig_t lc = {};
    lc.gridDim  = dim3(cs*NCL,1,1);
    lc.blockDim = dim3(128,1,1);
    cudaLaunchAttribute at{};
    at.id = cudaLaunchAttributeClusterDimension;
    at.val.clusterDim = {(unsigned)cs,1,1};
    lc.attrs=&at; lc.numAttrs=1;
    cudaError_t e = cudaLaunchKernelEx(&lc, cluster_probe, dout, sentinel);
    if (e != cudaSuccess) { printf("[cs=%2d] LAUNCH FAILED: %s\n", cs, cudaGetErrorString(e)); cudaGetLastError(); continue; }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("[cs=%2d] SYNC FAILED: %s\n", cs, cudaGetErrorString(e)); cudaGetLastError(); continue; }
    CK(cudaMemcpy(hout, dout, cs*NCL*4*sizeof(int), cudaMemcpyDeviceToHost));
    bool dsm_ok=true, chain_ok=true, store_ok=true;
    for (int b=0;b<cs*NCL;b++){
      int rank=b%cs, prev=(rank-1+cs)%cs;
      if (hout[b*4+0]!=cs)          dsm_ok=false;      // num_blocks
      if (hout[b*4+1]!=sentinel)    dsm_ok=false;      // DSM read of rank0
      if (hout[b*4+2]!=7000+prev)   chain_ok=false;    // producer/consumer
      if (hout[b*4+3]!=9000+prev)   store_ok=false;    // remote store landed
    }
    // rank 1 is block index 1 in cluster 0; its prev rank is 0 -> want chain=7000, store=9000
    printf("[cs=%2d] LAUNCH OK | DSM=%s cluster.sync/chain=%s remote_store=%s | rank1 sample: n=%d(want %d) dsm=%d(want %d) chain=%d(want 7000) store=%d(want 9000)\n",
      cs, dsm_ok?"PASS":"FAIL", chain_ok?"PASS":"FAIL", store_ok?"PASS":"FAIL",
      hout[4],cs, hout[5],sentinel, hout[6], hout[7]);
    if (dsm_ok&&chain_ok&&store_ok) largest_ok = cs;
  }
  printf("LARGEST_CLUSTER_LAUNCHED = %d\n", largest_ok);

  // (4b) Barrier race demonstrator at cs=4 --------------------------------------
  {
    int cs=4, sentinel=999000;
    CK(cudaMemset(dout,0,cs*sizeof(int)));
    cudaLaunchConfig_t lc={};
    lc.gridDim=dim3(cs,1,1); lc.blockDim=dim3(128,1,1);
    cudaLaunchAttribute at{}; at.id=cudaLaunchAttributeClusterDimension;
    at.val.clusterDim={(unsigned)cs,1,1};
    lc.attrs=&at; lc.numAttrs=1;
    cudaError_t e=cudaLaunchKernelEx(&lc, cluster_probe_race, dout, sentinel);
    if(e==cudaSuccess) e=cudaDeviceSynchronize();
    if(e==cudaSuccess){
      CK(cudaMemcpy(hout,dout,cs*sizeof(int),cudaMemcpyDeviceToHost));
      int mism=0; for(int b=1;b<cs;b++) if(hout[b]!=sentinel) mism++;
      printf("[race] NO-barrier DSM read of delayed rank0 smem: ranks1-3 = %d,%d,%d (sentinel=%d); %d/%d peers read STALE -> barrier is load-bearing%s\n",
        hout[1],hout[2],hout[3],sentinel,mism,cs-1, mism? "":" (no mismatch this run = timing luck, not a barrier bug)");
    } else { printf("[race] launch/sync error: %s\n", cudaGetErrorString(e)); cudaGetLastError(); }
  }

  printf("CLUSTER_PROBE_DONE\n");
  return 0;
}
