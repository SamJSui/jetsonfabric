package election

import (
	"sort"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/membership"
)

func ElectLeader(now time.Time, members []membership.Member, staleAfter time.Duration) (membership.Member, bool) {
	peers := activePeers(now, members, staleAfter)
	if len(peers) == 0 {
		return membership.Member{}, false
	}
	sort.SliceStable(peers, func(i, j int) bool {
		return betterPeer(peers[i], peers[j])
	})
	return peers[0], true
}

func activePeers(now time.Time, members []membership.Member, staleAfter time.Duration) []membership.Member {
	peers := make([]membership.Member, 0, len(members))
	for _, member := range members {
		member = membership.Normalize(member)
		if allowedPeer(now, member, staleAfter) {
			peers = append(peers, member)
		}
	}
	return peers
}

func allowedPeer(now time.Time, member membership.Member, staleAfter time.Duration) bool {
	if !member.Valid() || member.IsStale(now, staleAfter) {
		return false
	}
	return membership.RoleRank(member.EffectiveRole()) > 0
}

func betterPeer(left membership.Member, right membership.Member) bool {
	leftRank := membership.RoleRank(left.EffectiveRole())
	rightRank := membership.RoleRank(right.EffectiveRole())
	if leftRank != rightRank {
		return leftRank > rightRank
	}
	return olderOrStable(left, right)
}

func olderOrStable(left membership.Member, right membership.Member) bool {
	if !left.StartedAt.Equal(right.StartedAt) {
		return startedEarlier(left.StartedAt, right.StartedAt)
	}
	return left.NodeID < right.NodeID
}

func startedEarlier(left time.Time, right time.Time) bool {
	if left.IsZero() {
		return false
	}
	if right.IsZero() {
		return true
	}
	return left.Before(right)
}
