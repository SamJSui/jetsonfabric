package node

import (
	"github.com/SamJSui/jetsonfabric/internal/modelartifacts"
)

// computeModelArtifactSHA256 computes the identity used to prove that every
// pipeline stage loaded the same model artifact. It streams the file instead of
// reading a potentially multi-gigabyte GGUF into memory.
func computeModelArtifactSHA256(path string) (string, error) {
	return modelartifacts.ComputeSHA256(path)
}
