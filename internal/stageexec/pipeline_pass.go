package stageexec

import "context"

// RunPipelinePass performs one ordered traversal of the planned model stages.
// A pass is either prefill or one decode step. Generate owns the higher-level
// completion lifecycle and invokes this method once per required forward pass.
func (e *Executor) RunPipelinePass(ctx context.Context, req Request) (Result, error) {
	return e.Execute(ctx, req)
}
