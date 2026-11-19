// FlashLib vibecuda k-means backend — TEST-ONLY probe extension (SM100).
//
// Exposes kda_lloyd_into, a deterministic caller-owned-output variant of the
// production pipeline (same kernels, same launch schedule; only the final
// assign/update destinations are caller-provided buffers). Kept in its own
// translation unit and shared object so it cannot perturb production code
// generation or loading.

#include "kmeans_lloyd_impl_sm100.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kda_lloyd_into", &kda_lloyd_into,
        "KDA Lloyd into caller-owned output buffers (testing only)");
}
