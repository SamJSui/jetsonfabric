package layersplit

import (
	"context"
	"sync"
	"time"
)

type LocalTransport struct {
	mu       sync.Mutex
	Requests []ActivationRequest
	Handler  func(StageTarget, ActivationRequest) (ActivationResponse, error)
}

func (t *LocalTransport) RunStage(_ context.Context, target StageTarget, req ActivationRequest) (ActivationResponse, error) {
	t.mu.Lock()
	t.Requests = append(t.Requests, req)
	t.mu.Unlock()

	start := time.Now()
	if t.Handler != nil {
		resp, err := t.Handler(target, req)
		if err != nil {
			return ActivationResponse{}, err
		}
		return NormalizeResponse(req, resp, time.Since(start)), nil
	}
	return SyntheticStageResponse(req, target.NodeName, time.Since(start)), nil
}
