package stageexec

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/inference"
	"github.com/SamJSui/jetsonfabric/internal/stagewire"
)

const sessionCleanupTimeout = 10 * time.Second

// Generate drives one prefill pass followed by decode passes over the same
// immutable plan and session ID. Every runtime retains its own stage-local model
// state between passes. The session is closed on every stage before Generate
// returns, including failure and cancellation paths.
func (e *Executor) Generate(ctx context.Context, req Request) (result Result, err error) {
	if e == nil {
		e = New(Config{})
	}
	maxTokens := normalizeGenerationMaxTokens(req.MaxTokens)

	prefillReq := req
	prefillReq.Phase = inference.PhasePrefill
	prefillReq.DecodeStep = 0
	prefillReq.Kind = stagewire.PayloadKindText
	prefillReq.Data = nil
	prefillReq.StrictPayloadTransitions = true
	prefillReq.MaxTokens = maxTokens

	pass, err := e.Execute(ctx, prefillReq)
	sessionID := pass.RequestID
	if sessionID == "" {
		sessionID = strings.TrimSpace(prefillReq.RequestID)
	}
	defer func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), sessionCleanupTimeout)
		defer cancel()
		cleanupErr := e.CloseSession(cleanupCtx, sessionID, req.Model, req.Plan)
		if cleanupErr != nil {
			if err == nil {
				err = cleanupErr
			} else {
				err = errors.Join(err, cleanupErr)
			}
		}
	}()
	if err != nil {
		return pass, err
	}
	result = emptyGenerationResult(pass)

	if err := mergeGenerationPass(&result, pass); err != nil {
		return result, err
	}
	for decodeStep := 1; len(result.SampledTokens) < maxTokens && !result.EndOfGeneration; decodeStep++ {
		previous := result.SampledTokens[len(result.SampledTokens)-1]
		payload := make([]byte, 4)
		binary.LittleEndian.PutUint32(payload, previous)
		decodeReq := Request{
			RequestID:                sessionID,
			Model:                    req.Model,
			Data:                     payload,
			Kind:                     stagewire.PayloadKindSampledToken,
			Phase:                    inference.PhaseDecode,
			DecodeStep:               decodeStep,
			MaxTokens:                maxTokens,
			Plan:                     req.Plan,
			StrictPayloadTransitions: true,
		}
		pass, err = e.Execute(ctx, decodeReq)
		if err != nil {
			return result, err
		}
		if err := mergeGenerationPass(&result, pass); err != nil {
			return result, err
		}
	}
	if result.EndOfGeneration {
		result.FinishReason = "stop"
	} else {
		result.FinishReason = "length"
	}
	return result, nil
}

func emptyGenerationResult(pass Result) Result {
	return Result{
		RequestID: pass.RequestID,
		Model:     pass.Model,
		Stages:    make([]StageTrace, 0, len(pass.Stages)),
	}
}

func mergeGenerationPass(result *Result, pass Result) error {
	if pass.PayloadKind != stagewire.PayloadKindSampledToken || pass.SampledToken == nil {
		return fmt.Errorf("generation pass returned %q instead of sampled_token", pass.PayloadKind)
	}
	result.PayloadKind = pass.PayloadKind
	result.PayloadBytes = pass.PayloadBytes
	result.SampledToken = pass.SampledToken
	result.TokenText = pass.TokenText
	result.EndOfGeneration = pass.EndOfGeneration
	result.Data = append(result.Data[:0], pass.Data...)
	result.GeneratedText += pass.TokenText
	result.SampledTokens = append(result.SampledTokens, *pass.SampledToken)
	result.PromptTokens += pass.PromptTokens
	result.CompletionTokens += pass.CompletionTokens
	result.BytesIn += pass.BytesIn
	result.BytesOut += pass.BytesOut
	result.Stages = append(result.Stages, pass.Stages...)
	return nil
}

// CloseSession releases stage-local state on every runtime in the plan. It
// attempts all stages even when one stage returns an error.
func (e *Executor) CloseSession(ctx context.Context, sessionID string, model string, plan clusterplan.RoutePreview) error {
	if e == nil {
		e = New(Config{})
	}
	if strings.TrimSpace(sessionID) == "" {
		return fmt.Errorf("session ID is required")
	}
	var closeErrors []error
	for _, stage := range plan.Stages {
		response, status, _, callErr := e.callStage(ctx, stage, buildCloseSessionRequest(sessionID, model, stage))
		if callErr != nil {
			closeErrors = append(closeErrors, fmt.Errorf("close stage %d session: %w", stage.StageIndex, callErr))
			continue
		}
		if status < 200 || status >= 300 {
			closeErrors = append(closeErrors, StageError{
				StageIndex: stage.StageIndex,
				StatusCode: status,
				Code:       response.Error,
				Message:    response.Message,
			})
		}
	}
	return errors.Join(closeErrors...)
}

func normalizeGenerationMaxTokens(value int) int {
	if value <= 0 {
		return 128
	}
	if value > 1024 {
		return 1024
	}
	return value
}
