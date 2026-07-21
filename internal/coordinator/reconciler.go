package coordinator

import (
	"context"
	"errors"
	"log"
	"time"
)

// NotifyMembershipChanged coalesces membership refreshes. Reconciliation also
// retries periodically so a transient runtime or network failure can recover
// without another topology change.
func (s *Server) NotifyMembershipChanged() {
	select {
	case s.reconcileCh <- struct{}{}:
	default:
	}
}

func (s *Server) RunReconciler(ctx context.Context) {
	ticker := time.NewTicker(s.reconcileInterval)
	defer ticker.Stop()
	s.NotifyMembershipChanged()
	for {
		select {
		case <-ctx.Done():
			return
		case <-s.reconcileCh:
			s.runReconcileAttempt(ctx)
		case <-ticker.C:
			s.runReconcileAttempt(ctx)
		}
	}
}

func (s *Server) runReconcileAttempt(parent context.Context) {
	ctx, cancel := context.WithTimeout(parent, s.transitionTimeout)
	defer cancel()
	if err := s.Reconcile(ctx); err != nil && !errors.Is(err, context.Canceled) {
		log.Printf("deployment reconciliation pending: %v", err)
	}
}

// Reconcile computes a desired epoch from the last successful deployment
// intent and the current membership snapshot.
func (s *Server) Reconcile(ctx context.Context) error {
	s.reconcileMu.Lock()
	defer s.reconcileMu.Unlock()
	if s.isLeader != nil && !s.isLeader(s.now()) {
		return nil
	}
	pendingCleanupErr := s.retryDraining(ctx)
	intent, ok := s.deployments.activeIntent()
	if !ok {
		return pendingCleanupErr
	}
	_, cleanupErr, err := s.switchDeployment(ctx, intent.switchRequest(), false)
	if err == nil {
		result := errors.Join(pendingCleanupErr, cleanupErr)
		s.deployments.recordReconcileError(result, true)
		return result
	}
	snapshot := s.deployments.snapshot()
	healthy := snapshot.Active != nil && activePlanHealthy(
		*snapshot.Active,
		s.memberSource.List(),
		s.now(),
		s.memberStaleAfter,
	)
	result := errors.Join(pendingCleanupErr, err)
	s.deployments.recordReconcileError(result, healthy)
	return result
}

func (s *Server) retryDraining(ctx context.Context) error {
	snapshot := s.deployments.snapshot()
	var failures []error
	for _, plan := range snapshot.Draining {
		if err := s.retirePlan(ctx, plan); err != nil {
			failures = append(failures, err)
		}
	}
	return errors.Join(failures...)
}
