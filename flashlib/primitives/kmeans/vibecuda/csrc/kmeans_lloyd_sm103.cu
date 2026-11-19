// FlashLib vibecuda k-means backend — production extension (SM103/B300).
//
// Compiles the shared implementation header kmeans_lloyd_impl_sm103.cuh with
// a single PYBIND binding for the allocating entry point. The test-only probe
// extension (kmeans_lloyd_probe_sm103.cu) compiles the same header, so the
// executed pipeline is line-identical between the two translation units.

#include "kmeans_lloyd_impl_sm103.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kda_lloyd", &kda_lloyd,
        "KDA Lloyd clustering; iterations<=0 runs a single assignment pass");
}
