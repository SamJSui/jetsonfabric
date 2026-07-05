package discovery

import (
	"context"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/membership"
)

const (
	ModeStatic = "static"
	ModeMDNS   = "mdns"
	ModeNone   = "none"

	DefaultMDNSService       = "_jetsonfabric._tcp"
	DefaultMDNSDomain        = "local."
	DefaultMDNSBrowseTimeout = 2 * time.Second
)

type Source interface {
	Discover(ctx context.Context) ([]membership.Member, error)
}

type SelfFunc func() membership.Member
