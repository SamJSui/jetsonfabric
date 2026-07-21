package coordinator

import (
	"context"
	"errors"
	"sort"
	"sync"

	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
)

var (
	errDeploymentTransitioning = errors.New("deployment transition is in progress")
	errDeploymentPlanInvalid   = errors.New("deployment plan is invalid")
	errModelNotActive          = errors.New("requested model is not active")
	errDeploymentUnavailable   = errors.New("coordinator deployment is unavailable")
)

type deploymentPhase string

const (
	deploymentPhaseUnmanaged deploymentPhase = "unmanaged"
	deploymentPhasePreparing deploymentPhase = "preparing"
	deploymentPhaseActive    deploymentPhase = "active"
	deploymentPhaseDraining  deploymentPhase = "draining"
	deploymentPhaseDegraded  deploymentPhase = "degraded"
	deploymentPhaseFailed    deploymentPhase = "failed"
)

type deploymentSnapshot struct {
	Phase           deploymentPhase
	Active          *clusterplan.DeploymentPlan
	Preparing       *clusterplan.DeploymentPlan
	Draining        []clusterplan.DeploymentPlan
	InFlight        int
	InFlightByEpoch map[uint64]int
	LastError       string
	ProposedEpoch   uint64
}

type deploymentAdmission struct {
	Plan    *clusterplan.DeploymentPlan
	release func()
	once    *sync.Once
}

func (a deploymentAdmission) Release() {
	if a.release != nil && a.once != nil {
		a.once.Do(a.release)
	}
}

type deploymentState struct {
	mu        sync.Mutex
	changed   chan struct{}
	phase     deploymentPhase
	active    *clusterplan.DeploymentPlan
	preparing *clusterplan.DeploymentPlan
	draining  map[uint64]*clusterplan.DeploymentPlan
	inFlight  map[uint64]int
	intent    *deploymentIntent
	lastError string
	lastEpoch uint64
}

func newDeploymentState() *deploymentState {
	return &deploymentState{
		changed:  make(chan struct{}),
		phase:    deploymentPhaseUnmanaged,
		draining: make(map[uint64]*clusterplan.DeploymentPlan),
		inFlight: make(map[uint64]int),
	}
}

func (s *deploymentState) snapshot() deploymentSnapshot {
	s.mu.Lock()
	defer s.mu.Unlock()

	snapshot := deploymentSnapshot{
		Phase:           s.phase,
		Active:          copyPlan(s.active),
		Preparing:       copyPlan(s.preparing),
		InFlightByEpoch: make(map[uint64]int, len(s.inFlight)),
		LastError:       s.lastError,
		ProposedEpoch:   s.lastEpoch + 1,
	}
	for epoch, count := range s.inFlight {
		snapshot.InFlight += count
		snapshot.InFlightByEpoch[epoch] = count
	}
	for _, plan := range s.draining {
		snapshot.Draining = append(snapshot.Draining, *copyPlan(plan))
	}
	sortPlansByEpoch(snapshot.Draining)
	return snapshot
}

func (s *deploymentState) admit(modelID string) (deploymentAdmission, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.phase == deploymentPhaseDegraded || s.phase == deploymentPhaseFailed {
		return deploymentAdmission{}, errDeploymentUnavailable
	}
	if s.active != nil && s.active.Model().ModelID != modelID {
		return deploymentAdmission{}, errModelNotActive
	}

	epoch := uint64(0)
	plan := copyPlan(s.active)
	if plan != nil {
		epoch = plan.Identity().Epoch
	}
	s.inFlight[epoch]++
	return deploymentAdmission{
		Plan: plan,
		once: &sync.Once{},
		release: func() {
			s.release(epoch)
		},
	}, nil
}

func (s *deploymentState) beginTransition(next clusterplan.DeploymentPlan) (*clusterplan.DeploymentPlan, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.preparing != nil {
		return nil, errDeploymentTransitioning
	}
	if next.Identity().Epoch != s.lastEpoch+1 {
		return nil, errors.New("deployment epoch changed while the plan was being built")
	}

	s.lastEpoch = next.Identity().Epoch
	s.preparing = copyPlan(&next)
	s.phase = deploymentPhasePreparing
	s.lastError = ""
	s.signalLocked()
	return copyPlan(s.active), nil
}

func (s *deploymentState) publish(plan clusterplan.DeploymentPlan, intent deploymentIntent) *clusterplan.DeploymentPlan {
	s.mu.Lock()
	defer s.mu.Unlock()

	previous := copyPlan(s.active)
	if previous != nil {
		epoch := previous.Identity().Epoch
		s.draining[epoch] = copyPlan(previous)
	}
	s.active = copyPlan(&plan)
	s.preparing = nil
	intentCopy := intent
	s.intent = &intentCopy
	s.lastError = ""
	if previous == nil {
		s.phase = deploymentPhaseActive
	} else {
		s.phase = deploymentPhaseDraining
	}
	s.signalLocked()
	return previous
}

func (s *deploymentState) rollback(err error, healthy bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.preparing = nil
	s.setErrorLocked(err)
	s.setServingPhaseLocked(healthy)
	s.signalLocked()
}

func (s *deploymentState) recordReconcileError(err error, healthy bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.setErrorLocked(err)
	if s.preparing == nil && len(s.draining) == 0 {
		s.setServingPhaseLocked(healthy)
	}
	s.signalLocked()
}

func (s *deploymentState) waitForEpoch(ctx context.Context, epoch uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for s.inFlight[epoch] > 0 {
		changed := s.changed
		s.mu.Unlock()
		select {
		case <-ctx.Done():
			s.mu.Lock()
			return ctx.Err()
		case <-changed:
			s.mu.Lock()
		}
	}
	return nil
}

func (s *deploymentState) finishDraining(epoch uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.draining, epoch)
	if s.inFlight[epoch] == 0 {
		delete(s.inFlight, epoch)
	}
	if len(s.draining) == 0 && s.preparing == nil && s.active != nil {
		s.phase = deploymentPhaseActive
		s.lastError = ""
	}
	s.signalLocked()
}

func (s *deploymentState) activeIntent() (deploymentIntent, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.intent == nil || s.active == nil {
		return deploymentIntent{}, false
	}
	return *s.intent, true
}

func (s *deploymentState) release(epoch uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.inFlight[epoch] > 0 {
		s.inFlight[epoch]--
		if s.inFlight[epoch] == 0 && !s.epochTrackedLocked(epoch) {
			delete(s.inFlight, epoch)
		}
		s.signalLocked()
	}
}

func (s *deploymentState) epochTrackedLocked(epoch uint64) bool {
	if s.active != nil && s.active.Identity().Epoch == epoch {
		return true
	}
	if s.preparing != nil && s.preparing.Identity().Epoch == epoch {
		return true
	}
	_, ok := s.draining[epoch]
	return ok
}

func (s *deploymentState) setServingPhaseLocked(healthy bool) {
	switch {
	case s.active == nil:
		s.phase = deploymentPhaseFailed
	case !healthy:
		s.phase = deploymentPhaseDegraded
	case len(s.draining) > 0:
		s.phase = deploymentPhaseDraining
	default:
		s.phase = deploymentPhaseActive
	}
}

func (s *deploymentState) setErrorLocked(err error) {
	if err == nil {
		s.lastError = ""
		return
	}
	s.lastError = err.Error()
}

func (s *deploymentState) signalLocked() {
	close(s.changed)
	s.changed = make(chan struct{})
}

func copyPlan(plan *clusterplan.DeploymentPlan) *clusterplan.DeploymentPlan {
	if plan == nil {
		return nil
	}
	copy := *plan
	return &copy
}

func sortPlansByEpoch(plans []clusterplan.DeploymentPlan) {
	sort.Slice(plans, func(left, right int) bool {
		return plans[left].Identity().Epoch < plans[right].Identity().Epoch
	})
}
