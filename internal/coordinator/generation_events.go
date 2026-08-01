package coordinator

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/SamJSui/jetsonfabric/internal/chat"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
)

type runtimeGenerationResult struct {
	GeneratedText             string
	SampledTokens             []uint32
	FinishReason              string
	PromptTokens              int
	CompletionTokens          int
	StageCalls                int
	RemoteStageCalls          int
	RollbackStageCalls        int
	RemoteRollbackStageCalls  int
	RollbackStageUS           int64
	RollbackRemoteCallUS      int64
	BytesIn                   int64
	BytesOut                  int64
	StageTimings              []chat.StageTiming
	TargetDecodePasses        int
	SpeculativeDraftTokens    int
	SpeculativeAcceptedTokens int
}

type generationEventConsumer struct {
	expectedStages int
	result         runtimeGenerationResult
}

func consumeGenerationEvents(reader io.Reader, expectedStages int, onToken func(runtimebridge.GenerationEvent) error) (runtimeGenerationResult, error) {
	consumer := newGenerationEventConsumer(expectedStages)
	return consumeDecodedGenerationEvents(json.NewDecoder(reader), consumer, onToken)
}

func newGenerationEventConsumer(expectedStages int) *generationEventConsumer {
	return &generationEventConsumer{
		expectedStages: expectedStages,
		result:         runtimeGenerationResult{SampledTokens: make([]uint32, 0)},
	}
}

func consumeDecodedGenerationEvents(decoder *json.Decoder, consumer *generationEventConsumer, onToken func(runtimebridge.GenerationEvent) error) (runtimeGenerationResult, error) {
	for {
		event, err := decodeGenerationEvent(decoder)
		if err != nil {
			return consumer.result, err
		}
		done, err := consumer.accept(event)
		if err != nil {
			return consumer.result, err
		}
		if event.Type == "token" && onToken != nil {
			if err := onToken(event); err != nil {
				return consumer.result, err
			}
		}
		if done {
			return consumer.result, nil
		}
	}
}

func decodeGenerationEvent(decoder *json.Decoder) (runtimebridge.GenerationEvent, error) {
	var event runtimebridge.GenerationEvent
	if err := decoder.Decode(&event); err != nil {
		if err == io.EOF {
			return event, fmt.Errorf("runtime generation stream ended before a done event")
		}
		return event, fmt.Errorf("decode runtime generation event: %w", err)
	}
	return event, nil
}

func (c *generationEventConsumer) accept(event runtimebridge.GenerationEvent) (bool, error) {
	switch event.Type {
	case "token":
		if event.Token == nil || event.Index != len(c.result.SampledTokens) {
			return false, fmt.Errorf("runtime emitted an invalid token event at index %d", event.Index)
		}
		c.result.SampledTokens = append(c.result.SampledTokens, *event.Token)
		c.result.GeneratedText += event.Text
		return false, nil
	case "done":
		if event.FinishReason != "stop" && event.FinishReason != "length" {
			return false, fmt.Errorf("runtime emitted invalid finish_reason %q", event.FinishReason)
		}
		if len(event.SampledTokens) != len(c.result.SampledTokens) {
			return false, fmt.Errorf("runtime done event sampled-token count does not match token events")
		}
		for index := range event.SampledTokens {
			if event.SampledTokens[index] != c.result.SampledTokens[index] {
				return false, fmt.Errorf("runtime done event sampled token %d does not match token stream", index)
			}
		}
		if event.PromptTokens < 0 {
			return false, fmt.Errorf("runtime emitted invalid prompt token count %d", event.PromptTokens)
		}
		if event.CompletionTokens != len(c.result.SampledTokens) {
			return false, fmt.Errorf(
				"runtime completion token count was %d, want %d",
				event.CompletionTokens,
				len(c.result.SampledTokens),
			)
		}
		passes := 1 + event.TargetDecodePasses
		if event.TargetDecodePasses == 0 {
			passes = len(c.result.SampledTokens)
			if event.FinishReason == "stop" {
				passes++
			}
		}
		expectedCalls := passes * c.expectedStages
		expectedRemoteCalls := passes * (c.expectedStages - 1)
		if event.StageCalls != expectedCalls || event.RemoteStageCalls != expectedRemoteCalls {
			return false, fmt.Errorf(
				"runtime stage call accounting was stage_calls=%d remote_stage_calls=%d, want %d and %d",
				event.StageCalls, event.RemoteStageCalls, expectedCalls, expectedRemoteCalls,
			)
		}
		if event.RollbackStageCalls%c.expectedStages != 0 {
			return false, fmt.Errorf(
				"runtime rollback stage calls=%d is not divisible by stage count %d",
				event.RollbackStageCalls,
				c.expectedStages,
			)
		}
		expectedRemoteRollbacks := (event.RollbackStageCalls / c.expectedStages) *
			(c.expectedStages - 1)
		if event.RemoteRollbackStageCalls != expectedRemoteRollbacks {
			return false, fmt.Errorf(
				"runtime remote rollback stage calls=%d, want %d",
				event.RemoteRollbackStageCalls,
				expectedRemoteRollbacks,
			)
		}
		c.result.FinishReason = event.FinishReason
		c.result.PromptTokens = event.PromptTokens
		c.result.CompletionTokens = event.CompletionTokens
		c.result.StageCalls = event.StageCalls
		c.result.RemoteStageCalls = event.RemoteStageCalls
		c.result.RollbackStageCalls = event.RollbackStageCalls
		c.result.RemoteRollbackStageCalls = event.RemoteRollbackStageCalls
		c.result.RollbackStageUS = event.RollbackStageUS
		c.result.RollbackRemoteCallUS = event.RollbackRemoteCallUS
		c.result.BytesIn = event.BytesIn
		c.result.BytesOut = event.BytesOut
		c.result.StageTimings = append([]chat.StageTiming(nil), event.StageTimings...)
		c.result.TargetDecodePasses = event.TargetDecodePasses
		c.result.SpeculativeDraftTokens = event.SpeculativeDraftTokens
		c.result.SpeculativeAcceptedTokens = event.SpeculativeAcceptedTokens
		return true, nil
	case "error":
		return false, fmt.Errorf("%s: %s", event.Code, event.Message)
	default:
		return false, fmt.Errorf("runtime emitted unknown generation event type %q", event.Type)
	}
}

func runtimeTrace(result runtimeGenerationResult) *chat.RuntimeTrace {
	return &chat.RuntimeTrace{
		StageCalls:                result.StageCalls,
		RemoteStageCalls:          result.RemoteStageCalls,
		RollbackStageCalls:        result.RollbackStageCalls,
		RemoteRollbackStageCalls:  result.RemoteRollbackStageCalls,
		RollbackStageUS:           result.RollbackStageUS,
		RollbackRemoteCallUS:      result.RollbackRemoteCallUS,
		BytesIn:                   result.BytesIn,
		BytesOut:                  result.BytesOut,
		StageTimings:              append([]chat.StageTiming(nil), result.StageTimings...),
		TargetDecodePasses:        result.TargetDecodePasses,
		SpeculativeDraftTokens:    result.SpeculativeDraftTokens,
		SpeculativeAcceptedTokens: result.SpeculativeAcceptedTokens,
	}
}
