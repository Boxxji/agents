// Package swebench implements a reproducible SWE-bench Pro V2 evaluation
// harness with September 2026 anti-leakage workspace sanitization,
// ai:inference-only credential validation, and vfs-vs-NVMe IO reporting.
package swebench

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math/rand/v2"
	"regexp"
	"sort"
	"strings"
	"time"
)

const (
	// RevisionSep2026 identifies the September 2026 anti-leakage dataset revision.
	RevisionSep2026 = "sep-2026"
	// BenchmarkVersionV2 identifies SWE-bench Pro V2 tasks.
	BenchmarkVersionV2 = "v2"
)

var (
	// ErrEmptySubset indicates no tasks matched the subset filter criteria.
	ErrEmptySubset = errors.New("swebench: subset contains no matching tasks")
	// ErrInvalidSubsetSpec indicates a malformed or non-compliant subset manifest.
	ErrInvalidSubsetSpec = errors.New("swebench: invalid subset specification")

	hexCommitRE = regexp.MustCompile(`^[0-9a-fA-F]{7,64}$`)
)

// TaskSpec describes a single SWE-bench Pro V2 evaluation instance.
type TaskSpec struct {
	InstanceID       string            `json:"instance_id"`
	Repo             string            `json:"repo"`
	BaseCommit       string            `json:"base_commit"`
	Revision         string            `json:"revision"`
	Version          string            `json:"version"`
	Language         string            `json:"language"`
	Difficulty       string            `json:"difficulty,omitempty"`
	TestPatchHash    string            `json:"test_patch_hash,omitempty"`
	BaseTimestamp    time.Time         `json:"base_timestamp,omitempty"`
	FixTimestamp     time.Time         `json:"fix_timestamp,omitempty"`
	EnvironmentImage string            `json:"environment_image,omitempty"`
	Metadata         map[string]string `json:"metadata,omitempty"`
}

// Validate checks that a TaskSpec conforms to the sep-2026 v2 requirements.
func (t TaskSpec) Validate(cutoff time.Time) error {
	if strings.TrimSpace(t.InstanceID) == "" {
		return fmt.Errorf("%w: instance_id is required", ErrInvalidSubsetSpec)
	}
	if strings.TrimSpace(t.Repo) == "" {
		return fmt.Errorf("%w: task %s missing repo", ErrInvalidSubsetSpec, t.InstanceID)
	}
	if !hexCommitRE.MatchString(strings.TrimSpace(t.BaseCommit)) {
		return fmt.Errorf("%w: task %s has invalid base_commit %q", ErrInvalidSubsetSpec, t.InstanceID, t.BaseCommit)
	}
	if normalizeRevision(t.Revision) != RevisionSep2026 {
		return fmt.Errorf("%w: task %s revision %q != %q", ErrInvalidSubsetSpec, t.InstanceID, t.Revision, RevisionSep2026)
	}
	if normalizeVersion(t.Version) != BenchmarkVersionV2 {
		return fmt.Errorf("%w: task %s version %q != %q", ErrInvalidSubsetSpec, t.InstanceID, t.Version, BenchmarkVersionV2)
	}
	if !cutoff.IsZero() && !t.BaseTimestamp.IsZero() && t.BaseTimestamp.After(cutoff) {
		return fmt.Errorf("%w: task %s base_timestamp %s exceeds cutoff %s", ErrInvalidSubsetSpec, t.InstanceID, t.BaseTimestamp.Format(time.RFC3339), cutoff.Format(time.RFC3339))
	}
	return nil
}

// SubsetFilterOptions configures reproducible selection of sep-2026 v2 tasks.
type SubsetFilterOptions struct {
	Name       string    `json:"name"`
	Revision   string    `json:"revision,omitempty"`
	Version    string    `json:"version,omitempty"`
	CutoffDate time.Time `json:"cutoff_date,omitempty"`
	Languages  []string  `json:"languages,omitempty"`
	Repos      []string  `json:"repos,omitempty"`
	Difficulty []string  `json:"difficulty,omitempty"`
	MaxTasks   int       `json:"max_tasks,omitempty"`
	Seed       uint64    `json:"seed,omitempty"`
	Strict     bool      `json:"strict,omitempty"`
}

// SubsetSpec is a reproducible, content-addressed SWE-bench Pro V2 task subset.
type SubsetSpec struct {
	Name             string     `json:"name"`
	Revision         string     `json:"revision"`
	BenchmarkVersion string     `json:"benchmark_version"`
	CutoffDate       time.Time  `json:"cutoff_date,omitempty"`
	Seed             uint64     `json:"seed,omitempty"`
	ManifestDigest   string     `json:"manifest_digest"`
	Tasks            []TaskSpec `json:"tasks"`
}

// FilterSep2026V2 filters and orders candidate tasks into a deterministic sep-2026 v2 subset.
// When opts.Strict is true, any candidate violating the sep-2026 v2 revision or cutoff fails immediately;
// otherwise non-conforming candidates are filtered out cleanly.
func FilterSep2026V2(candidates []TaskSpec, opts SubsetFilterOptions) (*SubsetSpec, error) {
	rev := normalizeRevision(opts.Revision)
	if rev == "" {
		rev = RevisionSep2026
	}
	if rev != RevisionSep2026 {
		return nil, fmt.Errorf("%w: unsupported revision %q (expected %q)", ErrInvalidSubsetSpec, opts.Revision, RevisionSep2026)
	}

	ver := normalizeVersion(opts.Version)
	if ver == "" {
		ver = BenchmarkVersionV2
	}
	if ver != BenchmarkVersionV2 {
		return nil, fmt.Errorf("%w: unsupported version %q (expected %q)", ErrInvalidSubsetSpec, opts.Version, BenchmarkVersionV2)
	}

	if opts.MaxTasks < 0 {
		return nil, fmt.Errorf("%w: max_tasks cannot be negative", ErrInvalidSubsetSpec)
	}

	langSet := toLowerSet(opts.Languages)
	repoSet := toLowerSet(opts.Repos)
	diffSet := toLowerSet(opts.Difficulty)

	seen := make(map[string]struct{}, len(candidates))
	filtered := make([]TaskSpec, 0, len(candidates))
	for _, task := range candidates {
		if err := task.Validate(opts.CutoffDate); err != nil {
			if opts.Strict {
				return nil, err
			}
			continue
		}
		if _, dup := seen[task.InstanceID]; dup {
			return nil, fmt.Errorf("%w: duplicate instance_id %q", ErrInvalidSubsetSpec, task.InstanceID)
		}
		if len(langSet) > 0 && !inSet(langSet, task.Language) {
			continue
		}
		if len(repoSet) > 0 && !inSet(repoSet, task.Repo) {
			continue
		}
		if len(diffSet) > 0 && !inSet(diffSet, task.Difficulty) {
			continue
		}
		seen[task.InstanceID] = struct{}{}
		task.Revision = RevisionSep2026
		task.Version = BenchmarkVersionV2
		filtered = append(filtered, task)
	}

	if len(filtered) == 0 {
		return nil, ErrEmptySubset
	}

	// Sort canonically before optional deterministic shuffle so subset selection is
	// invariant to input slice order.
	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].InstanceID < filtered[j].InstanceID
	})

	if opts.Seed != 0 && len(filtered) > 1 {
		rng := rand.New(rand.NewPCG(opts.Seed, opts.Seed^0x9e3779b97f4a7c15))
		rng.Shuffle(len(filtered), func(i, j int) {
			filtered[i], filtered[j] = filtered[j], filtered[i]
		})
	}

	if opts.MaxTasks > 0 && len(filtered) > opts.MaxTasks {
		filtered = filtered[:opts.MaxTasks]
	}

	name := strings.TrimSpace(opts.Name)
	if name == "" {
		name = "swebench-pro-v2-sep-2026"
	}

	spec := &SubsetSpec{
		Name:             name,
		Revision:         RevisionSep2026,
		BenchmarkVersion: BenchmarkVersionV2,
		CutoffDate:       opts.CutoffDate.UTC(),
		Seed:             opts.Seed,
		Tasks:            filtered,
	}
	spec.ManifestDigest = computeManifestDigest(spec)
	return spec, nil
}

// Validate verifies subset integrity, sep-2026 v2 compliance, and manifest digest parity.
func (s *SubsetSpec) Validate() error {
	if s == nil {
		return fmt.Errorf("%w: subset is nil", ErrInvalidSubsetSpec)
	}
	if normalizeRevision(s.Revision) != RevisionSep2026 {
		return fmt.Errorf("%w: revision %q != %q", ErrInvalidSubsetSpec, s.Revision, RevisionSep2026)
	}
	if normalizeVersion(s.BenchmarkVersion) != BenchmarkVersionV2 {
		return fmt.Errorf("%w: benchmark_version %q != %q", ErrInvalidSubsetSpec, s.BenchmarkVersion, BenchmarkVersionV2)
	}
	if len(s.Tasks) == 0 {
		return ErrEmptySubset
	}

	seen := make(map[string]struct{}, len(s.Tasks))
	for _, task := range s.Tasks {
		if err := task.Validate(s.CutoffDate); err != nil {
			return err
		}
		if _, exists := seen[task.InstanceID]; exists {
			return fmt.Errorf("%w: duplicate instance_id %q", ErrInvalidSubsetSpec, task.InstanceID)
		}
		seen[task.InstanceID] = struct{}{}
	}

	expected := computeManifestDigest(s)
	if s.ManifestDigest != expected {
		return fmt.Errorf("%w: manifest digest mismatch (got %s, want %s)", ErrInvalidSubsetSpec, s.ManifestDigest, expected)
	}
	return nil
}

func computeManifestDigest(s *SubsetSpec) string {
	h := sha256.New()
	_, _ = fmt.Fprintf(h, "name=%s;rev=%s;ver=%s;seed=%d;cutoff=%d\n",
		s.Name, s.Revision, s.BenchmarkVersion, s.Seed, s.CutoffDate.UTC().Unix())
	for _, t := range s.Tasks {
		_, _ = fmt.Fprintf(h, "%s|%s|%s|%s|%s|%s\n",
			t.InstanceID, t.Repo, strings.ToLower(t.BaseCommit), t.Revision, t.Version, t.TestPatchHash)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func normalizeRevision(rev string) string {
	return strings.ToLower(strings.TrimSpace(rev))
}

func normalizeVersion(ver string) string {
	v := strings.ToLower(strings.TrimSpace(ver))
	if v == "2" || v == "pro-v2" {
		return BenchmarkVersionV2
	}
	return v
}

func toLowerSet(items []string) map[string]struct{} {
	if len(items) == 0 {
		return nil
	}
	set := make(map[string]struct{}, len(items))
	for _, item := range items {
		norm := strings.ToLower(strings.TrimSpace(item))
		if norm != "" {
			set[norm] = struct{}{}
		}
	}
	return set
}

func inSet(set map[string]struct{}, value string) bool {
	_, ok := set[strings.ToLower(strings.TrimSpace(value))]
	return ok
}
