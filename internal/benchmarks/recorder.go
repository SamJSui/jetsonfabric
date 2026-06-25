package benchmarks

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type Record struct {
	Timestamp    time.Time `json:"timestamp"`
	ModelID      string    `json:"model_id"`
	NodeID       string    `json:"node_id"`
	RouteMode    string    `json:"route_mode"`
	BackendID    string    `json:"backend_id"`
	BackendKind  string    `json:"backend_kind"`
	LatencyMS    int64     `json:"latency_ms"`
	OutputTokens int       `json:"output_tokens,omitempty"`
	TokensPerSec float64   `json:"tokens_per_sec,omitempty"`
	MemoryGB     *float64  `json:"memory_gb,omitempty"`
	TemperatureC *float64  `json:"temperature_c,omitempty"`
	Error        string    `json:"error,omitempty"`
}

type Recorder interface {
	Record(ctx context.Context, record Record) error
}

type NoopRecorder struct{}

func (NoopRecorder) Record(context.Context, Record) error {
	return nil
}

type JSONLRecorder struct {
	path string
	mu   sync.Mutex
}

func NewJSONLRecorder(path string) *JSONLRecorder {
	return &JSONLRecorder{path: path}
}

func (r *JSONLRecorder) Record(_ context.Context, record Record) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := os.MkdirAll(filepath.Dir(r.path), 0o755); err != nil {
		return fmt.Errorf("create benchmark directory: %w", err)
	}
	file, err := os.OpenFile(r.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("open benchmark file: %w", err)
	}
	defer file.Close()
	if err := json.NewEncoder(file).Encode(record); err != nil {
		return fmt.Errorf("write benchmark record: %w", err)
	}
	return nil
}
