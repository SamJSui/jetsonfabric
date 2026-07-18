package clusterplan

import "github.com/SamJSui/jetsonfabric/internal/cluster"

// PreviewPipeline returns a route that always uses the pipeline execution
// contract, including the degenerate one-stage case. Preview still owns member
// eligibility, explicit runtime assignment validation, layer coverage, topology,
// and placement policy. A valid one-stage full-model route is promoted from the
// legacy data_parallel label so callers can use one stage protocol for every
// cluster size.
func PreviewPipeline(req Request) RoutePreview {
	preview := Preview(req)
	if !preview.Valid || len(preview.Stages) != 1 || preview.StageCount != 1 {
		return preview
	}
	if !supportsMode(req.Model, cluster.ExecutionModePipelineParallel) {
		return preview
	}
	preview.Mode = cluster.ExecutionModePipelineParallel
	return preview
}
