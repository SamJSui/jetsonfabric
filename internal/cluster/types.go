package cluster

import "time"

type NodeRecord struct {
	NodeID       string         `json:"node_id"`
	Hostname     string         `json:"hostname"`
	Arch         string         `json:"arch"`
	OS           string         `json:"os"`
	Capabilities map[string]any `json:"capabilities"`
	Metrics      map[string]any `json:"metrics"`
	LastSeen     time.Time      `json:"last_seen"`
}

type HeartbeatRequest struct {
	NodeID       string         `json:"node_id"`
	Hostname     string         `json:"hostname"`
	Arch         string         `json:"arch"`
	OS           string         `json:"os"`
	Capabilities map[string]any `json:"capabilities"`
	Metrics      map[string]any `json:"metrics"`
}

type ModelProfile struct {
	ID                   string   `json:"id"`
	Family               string   `json:"family"`
	Runtime              string   `json:"runtime"`
	MinMemoryGB          float64  `json:"min_memory_gb"`
	PreferredAccelerator *string  `json:"preferred_accelerator"`
	PlacementModes       []string `json:"placement_modes"`
}
