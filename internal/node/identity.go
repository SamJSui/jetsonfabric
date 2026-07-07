package node

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	nodeIDFileName  = "node-id"
	defaultDataRoot = ".cache/jetsonfabric/nodes"
)

func PrepareInstanceConfig(cfg Config) (Config, error) {
	if strings.TrimSpace(cfg.NodeName) == "" {
		nodeName, err := defaultLogicalNodeName()
		if err != nil {
			return Config{}, err
		}
		cfg.NodeName = nodeName
	}

	if strings.TrimSpace(cfg.DataDir) == "" {
		cfg.DataDir = filepath.Join(defaultDataRoot, sanitizePathName(cfg.NodeName))
	}

	return cfg, nil
}

func defaultLogicalNodeName() (string, error) {
	hostname, _ := os.Hostname()
	hostname = strings.TrimSpace(strings.TrimSuffix(hostname, "."))
	if hostname == "" {
		hostname = "node"
	}

	suffix, err := randomHex(2)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%s-%s", hostname, suffix), nil
}

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

func randomHex(n int) (string, error) {
	bytes := make([]byte, n)
	if _, err := rand.Read(bytes); err != nil {
		return "", fmt.Errorf("generate random suffix: %w", err)
	}
	return hex.EncodeToString(bytes), nil
}

func sanitizePathName(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "node"
	}

	var b strings.Builder
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '-', r == '_', r == '.':
			b.WriteRune(r)
		default:
			b.WriteRune('-')
		}
	}

	out := strings.Trim(b.String(), "-._")
	if out == "" {
		return "node"
	}
	return out
}
