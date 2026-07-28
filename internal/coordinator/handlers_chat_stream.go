package coordinator

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
)

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
