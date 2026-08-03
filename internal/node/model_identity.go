package node

import (
	"github.com/SamJSui/jetsonfabric/internal/modelartifacts"
)

// computeModelArtifactSHA256 computes the source-model identity used to prove
// that every stage loaded the same artifact, whether GGUF or compiled JFM.
func computeModelArtifactSHA256(path string) (string, error) {
	return modelartifacts.ComputeSHA256(path)
}
