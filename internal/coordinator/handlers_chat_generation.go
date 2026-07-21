package coordinator

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
)

type runtimeGenerationResult struct {
	GeneratedText    string
	SampledTokens    []uint32
	FinishReason     string
	PromptTokens     int
	CompletionTokens int
	StageCalls       int
	RemoteStageCalls int
}

type generationEventConsumer struct {
	expectedStages int
	result         runtimeGenerationResult
}

type chatCompletionChunk struct {
	ID      string                      `json:"id"`
	Object  string                      `json:"object"`
	Created int64                       `json:"created"`
	Model   string                      `json:"model"`
	Choices []chatCompletionChunkChoice `json:"choices"`
}

type chatCompletionChunkChoice struct {
	Index        int                 `json:"index"`
	Delta        chatCompletionDelta `json:"delta"`
	FinishReason *string             `json:"finish_reason"`
}

type chatCompletionDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

func setGenerationHeaders(w http.ResponseWriter, sessionID string, plan clusterplan.RoutePreview, identity pipelineRuntimeIdentity) {
	w.Header().Set("X-JetsonFabric-Session-ID", sessionID)
	w.Header().Set("X-JetsonFabric-Topology", string(plan.Topology))
	w.Header().Set("X-JetsonFabric-Model-SHA256", identity.ModelSHA256)
	w.Header().Set("X-JetsonFabric-Generation-Owner", "runtime")
	w.Header().Set("X-JetsonFabric-Pipeline-Leader", plan.Stages[0].NodeID)
	if identity.DeploymentID != "" {
		w.Header().Set("X-JetsonFabric-Deployment-ID", identity.DeploymentID)
		w.Header().Set("X-JetsonFabric-Deployment-Epoch", fmt.Sprintf("%d", identity.Epoch))
	}
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
		passes := len(c.result.SampledTokens)
		if event.FinishReason == "stop" {
			passes++
		}
		expectedCalls := passes * c.expectedStages
		expectedRemoteCalls := passes * (c.expectedStages - 1)
		if event.StageCalls != expectedCalls || event.RemoteStageCalls != expectedRemoteCalls {
			return false, fmt.Errorf(
				"runtime stage call accounting was stage_calls=%d remote_stage_calls=%d, want %d and %d",
				event.StageCalls, event.RemoteStageCalls, expectedCalls, expectedRemoteCalls,
			)
		}
		c.result.FinishReason = event.FinishReason
		c.result.PromptTokens = event.PromptTokens
		c.result.CompletionTokens = event.CompletionTokens
		c.result.StageCalls = event.StageCalls
		c.result.RemoteStageCalls = event.RemoteStageCalls
		return true, nil
	case "error":
		return false, fmt.Errorf("%s: %s", event.Code, event.Message)
	default:
		return false, fmt.Errorf("runtime emitted unknown generation event type %q", event.Type)
	}
}

func (s *Server) streamChatCompletion(w http.ResponseWriter, r *http.Request, requestID string, modelID string, stageCount int, reader io.Reader) {
	decoder := json.NewDecoder(reader)
	consumer := newGenerationEventConsumer(stageCount)
	first, err := decodeGenerationEvent(decoder)
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, "server_error", "runtime_generation_failed", nil, err.Error())
		return
	}
	done, err := consumer.accept(first)
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, "server_error", "runtime_generation_failed", nil, err.Error())
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)

	if err := writeChatChunk(w, flusher, chatCompletionChunk{
		ID: requestID, Object: "chat.completion.chunk", Created: s.now().Unix(), Model: modelID,
		Choices: []chatCompletionChunkChoice{{Index: 0, Delta: chatCompletionDelta{Role: "assistant"}}},
	}); err != nil {
		return
	}
	writeToken := func(event runtimebridge.GenerationEvent) error {
		select {
		case <-r.Context().Done():
			return r.Context().Err()
		default:
		}
		return writeChatChunk(w, flusher, chatCompletionChunk{
			ID: requestID, Object: "chat.completion.chunk", Created: s.now().Unix(), Model: modelID,
			Choices: []chatCompletionChunkChoice{{Index: 0, Delta: chatCompletionDelta{Content: event.Text}}},
		})
	}
	if first.Type == "token" {
		if err := writeToken(first); err != nil {
			return
		}
	}
	result := consumer.result
	if !done {
		result, err = consumeDecodedGenerationEvents(decoder, consumer, writeToken)
		if err != nil {
			_ = writeSSEData(w, flusher, openAIErrorEnvelope{Error: openAIError{
				Message: err.Error(), Type: "server_error", Code: "runtime_generation_failed",
			}})
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			if flusher != nil {
				flusher.Flush()
			}
			return
		}
	}
	finishReason := result.FinishReason
	_ = writeChatChunk(w, flusher, chatCompletionChunk{
		ID: requestID, Object: "chat.completion.chunk", Created: s.now().Unix(), Model: modelID,
		Choices: []chatCompletionChunkChoice{{Index: 0, FinishReason: &finishReason}},
	})
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}

func writeChatChunk(w io.Writer, flusher http.Flusher, chunk chatCompletionChunk) error {
	return writeSSEData(w, flusher, chunk)
}

func writeSSEData(w io.Writer, flusher http.Flusher, value any) error {
	payload, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", payload); err != nil {
		return err
	}
	if flusher != nil {
		flusher.Flush()
	}
	return nil
}
