package coordinator

import (
	"context"
	"fmt"
	"io"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
)

type generationStartErrorKind string

const (
	generationUnknownModel          generationStartErrorKind = "unknown_model"
	generationMembershipUnavailable generationStartErrorKind = "membership_unavailable"
	generationIdentityUnavailable   generationStartErrorKind = "runtime_identity_unavailable"
	generationRouteUnavailable      generationStartErrorKind = "pipeline_route_unavailable"
	generationRuntimeUnavailable    generationStartErrorKind = "runtime_generation_failed"
)

type generationStartError struct {
	kind generationStartErrorKind
	err  error
}

func (e *generationStartError) Error() string {
	return e.err.Error()
}

type generationSpec struct {
	ModelID   string
	Prompt    string
	MaxTokens int
	Policy    clusterplan.Policy
}

type generationSession struct {
	RequestID string
	SessionID string
	Model     cluster.ModelProfile
	Plan      clusterplan.RoutePreview
	Identity  pipelineRuntimeIdentity
	Body      io.ReadCloser
	release   func()
}

func (s *generationSession) Close() {
	if s.Body != nil {
		_ = s.Body.Close()
	}
	if s.release != nil {
		s.release()
		s.release = nil
	}
}

// GenerationController pins deployment admission, selects stage zero, and
// starts runtime-owned generation. HTTP and OpenAI response formatting stay in
// Server.
type GenerationController struct {
	registry         modelRegistry
	memberSource     MemberSource
	memberStaleAfter time.Duration
	now              func() time.Time
	deployments      *DeploymentController
	runtimeClient    runtimebridge.GenerationClient
}

type modelRegistry interface {
	Find(string) (cluster.ModelProfile, bool)
}

func newGenerationController(
	registry modelRegistry,
	memberSource MemberSource,
	memberStaleAfter time.Duration,
	now func() time.Time,
	deployments *DeploymentController,
	runtimeClient runtimebridge.GenerationClient,
) *GenerationController {
	return &GenerationController{
		registry:         registry,
		memberSource:     memberSource,
		memberStaleAfter: memberStaleAfter,
		now:              now,
		deployments:      deployments,
		runtimeClient:    runtimeClient,
	}
}

func (c *GenerationController) Start(
	ctx context.Context,
	spec generationSpec,
) (*generationSession, error) {
	model, ok := c.registry.Find(spec.ModelID)
	if !ok {
		return nil, newGenerationStartError(
			generationUnknownModel,
			fmt.Errorf("model %q is not in the JetsonFabric registry", spec.ModelID),
		)
	}
	if c.memberSource == nil {
		return nil, newGenerationStartError(
			generationMembershipUnavailable,
			fmt.Errorf("membership source is required for pipeline chat completion"),
		)
	}

	admission, err := c.deployments.admit(spec.ModelID)
	if err != nil {
		return nil, err
	}
	releaseAdmission := true
	defer func() {
		if releaseAdmission {
			admission.Release()
		}
	}()

	plan, identity, err := c.resolvePlan(model, spec.Policy, admission)
	if err != nil {
		return nil, err
	}
	requestID := fmt.Sprintf("chatcmpl-%d", c.now().UnixNano())
	sessionID := fmt.Sprintf("session-%d", c.now().UnixNano())
	request := runtimeGenerationRequest(requestID, sessionID, model.ID, spec, plan, identity)

	stream, err := c.runtimeClient.Start(ctx, plan.Stages[0].APIURL, request)
	if err != nil {
		return nil, newGenerationStartError(generationRuntimeUnavailable, err)
	}

	releaseAdmission = false
	return &generationSession{
		RequestID: requestID,
		SessionID: sessionID,
		Model:     model,
		Plan:      plan,
		Identity:  identity,
		Body:      stream.Body,
		release:   admission.Release,
	}, nil
}

func (c *GenerationController) resolvePlan(
	model cluster.ModelProfile,
	policy clusterplan.Policy,
	admission deploymentAdmission,
) (clusterplan.RoutePreview, pipelineRuntimeIdentity, error) {
	if admission.Plan != nil {
		return admission.Plan.RoutePreview(), runtimeIdentityForDeployment(*admission.Plan), nil
	}
	requiredStages := policy.StageCount
	if requiredStages <= 0 {
		requiredStages = 1
		policy.StageCount = requiredStages
	}
	members, identity, err := selectPipelineRuntimeMembers(
		model,
		c.memberSource.List(),
		c.now(),
		c.memberStaleAfter,
		requiredStages,
	)
	if err != nil {
		return clusterplan.RoutePreview{}, pipelineRuntimeIdentity{},
			newGenerationStartError(generationIdentityUnavailable, err)
	}
	plan := clusterplan.PreviewPipeline(clusterplan.Request{
		Model:      model,
		Members:    members,
		Now:        c.now(),
		StaleAfter: c.memberStaleAfter,
		Policy:     policy,
	})
	if !plan.Valid || plan.Mode != cluster.ExecutionModePipelineParallel || plan.StageCount < 1 {
		return clusterplan.RoutePreview{}, pipelineRuntimeIdentity{}, newGenerationStartError(
			generationRouteUnavailable,
			fmt.Errorf("no valid pipeline route for model %q: %s", model.ID, plan.Reason),
		)
	}
	return plan, identity, nil
}

func runtimeGenerationRequest(
	requestID string,
	sessionID string,
	modelID string,
	spec generationSpec,
	plan clusterplan.RoutePreview,
	identity pipelineRuntimeIdentity,
) runtimebridge.GenerationRequest {
	request := runtimebridge.GenerationRequest{
		RequestID: requestID,
		SessionID: sessionID,
		ModelID:   modelID,
		Prompt:    spec.Prompt,
		MaxTokens: spec.MaxTokens,
		Stages:    plan.Stages,
	}
	if identity.DeploymentID != "" {
		request.Deployment = &runtimebridge.DeploymentIdentity{
			DeploymentID: identity.DeploymentID,
			Epoch:        identity.Epoch,
			ModelID:      identity.ModelID,
			ModelSHA256:  identity.ModelSHA256,
		}
	}
	return request
}

func newGenerationStartError(kind generationStartErrorKind, err error) error {
	return &generationStartError{kind: kind, err: err}
}
