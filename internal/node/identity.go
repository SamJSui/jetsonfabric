package node

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const nodeIDFileName = "node-id"

func LoadOrCreateNodeID(dataDir string) (string, error) {
	path := filepath.Join(dataDir, nodeIDFileName)
	content, err := os.ReadFile(path)
	if err == nil {
		nodeID := strings.TrimSpace(string(content))
		if nodeID != "" {
			return nodeID, nil
		}
	}
	if err != nil && !os.IsNotExist(err) {
		return "", fmt.Errorf("read node id: %w", err)
	}

	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return "", fmt.Errorf("create data dir: %w", err)
	}

	bytes := make([]byte, 16)
	if _, err := rand.Read(bytes); err != nil {
		return "", fmt.Errorf("generate node id: %w", err)
	}
	nodeID := hex.EncodeToString(bytes)
	if err := os.WriteFile(path, []byte(nodeID+"\n"), 0o600); err != nil {
		return "", fmt.Errorf("write node id: %w", err)
	}
	return nodeID, nil
}
