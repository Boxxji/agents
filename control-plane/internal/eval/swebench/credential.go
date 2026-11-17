package swebench

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

const (
	// ScopeAIInference is the only token scope permitted inside SWE-bench Visor sandboxes.
	ScopeAIInference = "ai:inference"
	// DefaultAPIKeyEnv is the canonical API key environment variable for inference calls.
	DefaultAPIKeyEnv = "HANZO_API_KEY"
)

// ErrCredentialPolicyViolation indicates a sandbox credential failed policy checks.
var ErrCredentialPolicyViolation = errors.New("swebench: sandbox credential policy violation")

// CredentialPolicy defines least-privilege credential constraints for sandboxed runs.
type CredentialPolicy struct {
	RequiredKeyEnv        string   `json:"required_key_env"`
	AllowedScopes         []string `json:"allowed_scopes"`
	ForbiddenEnvVars      []string `json:"forbidden_env_vars"`
	AllowedTenantID       string   `json:"allowed_tenant_id,omitempty"`
	RequireExplicitScopes bool     `json:"require_explicit_scopes"`
}

// DefaultSandboxCredentialPolicy enforces HANZO_API_KEY with strict ai:inference-only scope.
func DefaultSandboxCredentialPolicy(tenantID string) CredentialPolicy {
	return CredentialPolicy{
		RequiredKeyEnv:        DefaultAPIKeyEnv,
		AllowedScopes:         []string{ScopeAIInference},
		AllowedTenantID:       strings.TrimSpace(tenantID),
		RequireExplicitScopes: true,
		ForbiddenEnvVars: []string{
			"HANZO_ADMIN_KEY",
			"HANZO_BILLING_KEY",
			"HANZO_CLOUD_TOKEN",
			"HANZO_IAM_SECRET",
			"HANZO_AGENTS_API_KEY",
			"AWS_ACCESS_KEY_ID",
			"AWS_SECRET_ACCESS_KEY",
			"AWS_SESSION_TOKEN",
			"S3_ACCESS_KEY",
			"S3_SECRET_KEY",
			"GITHUB_TOKEN",
			"GH_TOKEN",
			"GIT_ASKPASS",
			"SSH_AUTH_SOCK",
			"DATABASE_URL",
		},
	}
}

// CredentialDescriptor describes the environment and scopes attached to a sandbox invocation.
type CredentialDescriptor struct {
	Env         map[string]string `json:"-"`
	TokenScopes []string          `json:"token_scopes"`
	TenantID    string            `json:"tenant_id,omitempty"`
	KeyID       string            `json:"key_id,omitempty"`
}

// CredentialAuditResult captures credential policy validation outcomes without leaking secret values.
type CredentialAuditResult struct {
	Valid             bool     `json:"valid"`
	TenantID          string   `json:"tenant_id,omitempty"`
	KeyID             string   `json:"key_id,omitempty"`
	KeyPreview        string   `json:"key_preview,omitempty"`
	GrantedScopes     []string `json:"granted_scopes,omitempty"`
	ForbiddenVarsSeen []string `json:"forbidden_vars_seen,omitempty"`
	Violations        []string `json:"violations,omitempty"`
}

// ValidateSandboxCredentials verifies that only the inference key with ai:inference
// scope is exposed and no privileged cloud, billing, git, or cross-tenant secrets exist.
func ValidateSandboxCredentials(desc CredentialDescriptor, policy CredentialPolicy) (*CredentialAuditResult, error) {
	if policy.RequiredKeyEnv == "" {
		policy.RequiredKeyEnv = DefaultAPIKeyEnv
	}
	if len(policy.AllowedScopes) == 0 {
		policy.AllowedScopes = []string{ScopeAIInference}
	}

	result := &CredentialAuditResult{
		TenantID: strings.TrimSpace(desc.TenantID),
		KeyID:    strings.TrimSpace(desc.KeyID),
	}

	if reqTenant := strings.TrimSpace(policy.AllowedTenantID); reqTenant != "" {
		if result.TenantID == "" || result.TenantID != reqTenant {
			result.Violations = append(result.Violations,
				fmt.Sprintf("tenant_id %q does not match required isolated tenant %q", result.TenantID, reqTenant))
		}
	}

	keyVal, hasKey := lookupEnvCaseInsensitive(desc.Env, policy.RequiredKeyEnv)
	if !hasKey || strings.TrimSpace(keyVal) == "" {
		result.Violations = append(result.Violations,
			fmt.Sprintf("missing required credential %s", policy.RequiredKeyEnv))
	} else if strings.ContainsAny(keyVal, "\r\n") {
		result.Violations = append(result.Violations,
			fmt.Sprintf("credential %s contains invalid control characters", policy.RequiredKeyEnv))
	} else {
		result.KeyPreview = redactSecret(keyVal)
	}

	allowedSet := toLowerSet(policy.AllowedScopes)
	normalizedScopes := make([]string, 0, len(desc.TokenScopes))
	for _, raw := range desc.TokenScopes {
		scope := strings.ToLower(strings.TrimSpace(raw))
		if scope == "" {
			continue
		}
		normalizedScopes = append(normalizedScopes, scope)
		if strings.Contains(scope, "*") {
			result.Violations = append(result.Violations,
				fmt.Sprintf("wildcard scope %q is forbidden in sandbox", raw))
			continue
		}
		if _, ok := allowedSet[scope]; !ok {
			result.Violations = append(result.Violations,
				fmt.Sprintf("scope %q exceeds allowed scopes %v", raw, policy.AllowedScopes))
		}
	}
	sort.Strings(normalizedScopes)
	result.GrantedScopes = normalizedScopes

	if policy.RequireExplicitScopes && len(normalizedScopes) == 0 {
		result.Violations = append(result.Violations,
			fmt.Sprintf("explicit %s scope is required", ScopeAIInference))
	} else if len(normalizedScopes) > 0 {
		if _, hasInference := toLowerSet(normalizedScopes)[ScopeAIInference]; !hasInference {
			result.Violations = append(result.Violations,
				fmt.Sprintf("required scope %q not granted", ScopeAIInference))
		}
	}

	forbiddenSet := toLowerSet(policy.ForbiddenEnvVars)
	for k, v := range desc.Env {
		if strings.TrimSpace(v) == "" {
			continue
		}
		if _, forbidden := forbiddenSet[strings.ToLower(strings.TrimSpace(k))]; forbidden {
			result.ForbiddenVarsSeen = append(result.ForbiddenVarsSeen, k)
			result.Violations = append(result.Violations,
				fmt.Sprintf("forbidden privileged environment variable %q present", k))
		}
	}
	sort.Strings(result.ForbiddenVarsSeen)
	sort.Strings(result.Violations)

	result.Valid = len(result.Violations) == 0
	if !result.Valid {
		return result, fmt.Errorf("%w: %s", ErrCredentialPolicyViolation, strings.Join(result.Violations, "; "))
	}
	return result, nil
}

// SanitizeSandboxEnv copies env while stripping all forbidden variables and empty values.
func SanitizeSandboxEnv(env map[string]string, policy CredentialPolicy) (map[string]string, []string) {
	forbiddenSet := toLowerSet(policy.ForbiddenEnvVars)
	clean := make(map[string]string, len(env))
	var stripped []string
	for k, v := range env {
		if _, forbidden := forbiddenSet[strings.ToLower(strings.TrimSpace(k))]; forbidden {
			stripped = append(stripped, k)
			continue
		}
		clean[k] = v
	}
	sort.Strings(stripped)
	return clean, stripped
}

func lookupEnvCaseInsensitive(env map[string]string, target string) (string, bool) {
	if val, ok := env[target]; ok {
		return val, true
	}
	for k, v := range env {
		if strings.EqualFold(strings.TrimSpace(k), target) {
			return v, true
		}
	}
	return "", false
}

func redactSecret(secret string) string {
	trimmed := strings.TrimSpace(secret)
	if len(trimmed) <= 8 {
		return "[REDACTED]"
	}
	return trimmed[:4] + "..." + trimmed[len(trimmed)-2:]
}
