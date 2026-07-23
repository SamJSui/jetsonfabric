package election

import (
	"sort"
	"sync"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/membership"
)

const (
	ReasonEligible        = "eligible"
	ReasonInvalidMember   = "invalid_member"
	ReasonStaleMember     = "stale_member"
	ReasonRoleNotEligible = "role_not_leader_eligible"
)

const (
	DefaultLeaseDuration = 10 * time.Second
	MaxLeaseDuration     = 15 * time.Second
)

type Result struct {
	Leader         *membership.Member `json:"leader,omitempty"`
	Candidates     []Candidate        `json:"candidates"`
	Epoch          uint64             `json:"epoch"`
	LeaseExpiresAt *time.Time         `json:"lease_expires_at,omitempty"`
}

type Candidate struct {
	Member           membership.Member   `json:"member"`
	Eligible         bool                `json:"eligible"`
	Reason           string              `json:"reason"`
	EffectiveRole    membership.NodeRole `json:"effective_role"`
	RoleRank         int                 `json:"role_rank"`
	LeaderPreference int                 `json:"leader_preference"`
}

type Tracker struct {
	mu            sync.Mutex
	state         State
	leaseDuration time.Duration
}

type State struct {
	LeaderID       string
	Epoch          uint64
	LeaseExpiresAt time.Time
}

func NewTracker(leaseDuration time.Duration) *Tracker {
	if leaseDuration <= 0 {
		leaseDuration = DefaultLeaseDuration
	}
	return &Tracker{leaseDuration: leaseDuration}
}

func DefaultLease(staleAfter time.Duration) time.Duration {
	if staleAfter <= 0 {
		return DefaultLeaseDuration
	}
	lease := staleAfter / 2
	if lease <= 0 {
		return DefaultLeaseDuration
	}
	if lease > MaxLeaseDuration {
		return MaxLeaseDuration
	}
	return lease
}

func ElectLeader(now time.Time, members []membership.Member, staleAfter time.Duration) (membership.Member, bool) {
	result := Explain(now, members, staleAfter)
	if result.Leader == nil {
		return membership.Member{}, false
	}
	return *result.Leader, true
}

func Explain(now time.Time, members []membership.Member, staleAfter time.Duration) Result {
	candidates := rankedCandidates(makeCandidates(now, members, staleAfter))
	return Result{Leader: firstEligible(candidates), Candidates: candidates}
}

func (t *Tracker) Explain(now time.Time, members []membership.Member, staleAfter time.Duration) Result {
	if t == nil {
		return Explain(now, members, staleAfter)
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	result := Explain(now, members, staleAfter)

	if result.Leader == nil {
		t.clearLeader(&result)
		return result
	}

	if t.state.LeaderID != result.Leader.NodeID {
		t.state.Epoch++
	}

	expiresAt := now.Add(t.leaseDuration)
	t.state.LeaderID = result.Leader.NodeID
	t.state.LeaseExpiresAt = expiresAt
	result.Epoch = t.state.Epoch
	result.LeaseExpiresAt = &expiresAt

	return result
}

func (t *Tracker) clearLeader(result *Result) {
	t.state.LeaderID = ""
	t.state.LeaseExpiresAt = time.Time{}
	result.Leader = nil
	result.Epoch = t.state.Epoch
	result.LeaseExpiresAt = nil
}

func makeCandidates(now time.Time, members []membership.Member, staleAfter time.Duration) []Candidate {
	candidates := make([]Candidate, 0, len(members))
	for _, member := range members {
		candidates = append(candidates, explainMember(now, member, staleAfter))
	}
	return candidates
}

func rankedCandidates(candidates []Candidate) []Candidate {
	sort.SliceStable(candidates, func(i, j int) bool {
		return betterCandidate(candidates[i], candidates[j])
	})
	return candidates
}

func explainMember(now time.Time, member membership.Member, staleAfter time.Duration) Candidate {
	member = membership.Normalize(member)
	candidate := newCandidate(member)
	if !member.Valid() {
		candidate.Reason = ReasonInvalidMember
		return candidate
	}
	if member.IsStale(now, staleAfter) {
		candidate.Reason = ReasonStaleMember
		return candidate
	}
	if candidate.RoleRank <= 0 {
		candidate.Reason = ReasonRoleNotEligible
		return candidate
	}
	candidate.Eligible = true
	candidate.Reason = ReasonEligible
	return candidate
}

func newCandidate(member membership.Member) Candidate {
	role := member.EffectiveRole()
	return Candidate{
		Member:           member,
		EffectiveRole:    role,
		RoleRank:         membership.RoleRank(role),
		LeaderPreference: member.EffectiveLeaderPreference(),
	}
}

func firstEligible(candidates []Candidate) *membership.Member {
	for _, candidate := range candidates {
		if candidate.Eligible {
			leader := candidate.Member
			return &leader
		}
	}
	return nil
}

func betterCandidate(left Candidate, right Candidate) bool {
	if left.Eligible != right.Eligible {
		return left.Eligible
	}
	if left.Eligible && right.Eligible {
		return betterPeer(left.Member, right.Member)
	}
	return left.Member.NodeID < right.Member.NodeID
}

func betterPeer(left membership.Member, right membership.Member) bool {
	leftRank := membership.RoleRank(left.EffectiveRole())
	rightRank := membership.RoleRank(right.EffectiveRole())
	if leftRank != rightRank {
		return leftRank > rightRank
	}
	if left.EffectiveLeaderPreference() != right.EffectiveLeaderPreference() {
		return left.EffectiveLeaderPreference() > right.EffectiveLeaderPreference()
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
