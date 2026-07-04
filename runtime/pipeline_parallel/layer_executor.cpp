#include "pipeline_parallel/layer_executor.hpp"

namespace jetsonfabric::runtime::pipeline_parallel {

StageRunResult UnavailableLayerExecutor::run_layers(const protocol::ActivationRequest& /*request*/) const {
    StageRunResult result;
    result.ok = false;
    result.status = "501 Not Implemented";
    result.error_code = "layer_executor_not_implemented";
    result.error_message = "stage request is valid, but transformer layer execution is not wired yet";
    return result;
}

} // namespace jetsonfabric::runtime::pipeline_parallel