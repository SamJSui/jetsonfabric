package election

import (
	"sort"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/membership"
)

// ElectLeader chooses the active coordinator from a local membership view.
//
// This is deterministic election, not consensus: every node that has converged
// on the same membership table will choose the same leader.
func ElectLeader(now time.Time, members []membership.Member, staleAfter time.Duration) (membership.Member, bool) {
	candidates := make([]membership.Member, 0, len(members))
	for _, member := range members {
		member = membership.Normalize(member)
		if !member.ControlEligible {
			continue
		}
		if !member.Valid() {
			continue
		}
		if member.IsStale(now, staleAfter) {
			continue
		}
		candidates = append(candidates, member)
	}

	if len(candidates) == 0 {
		return membership.Member{}, false
	}

	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].ControlPriority != candidates[j].ControlPriority {
			return candidates[i].ControlPriority > candidates[j].ControlPriority
		}
		return strings.Compare(candidates[i].NodeID, candidates[j].NodeID) < 0
	})

	return candidates[0], true
}
