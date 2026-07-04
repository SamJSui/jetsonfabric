package discovery

import (
	"context"

	"github.com/SamJSui/jetsonfabric/internal/membership"
)

type Source interface {
	Discover(ctx context.Context) ([]membership.Member, error)
}
