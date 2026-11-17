package swebench

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

var (
	// ErrLeakageDetected indicates a workspace failed anti-leakage or anti-tampering checks.
	ErrLeakageDetected = errors.New("swebench: workspace anti-leakage verification failed")
)

// ViolationCode categorizes workspace leakage or tampering findings.
type ViolationCode string

const (
	ViolationRemotePresent     ViolationCode = "remote_present"
	ViolationTagPresent        ViolationCode = "tag_present"
	ViolationExtraBranch       ViolationCode = "extra_branch_present"
	ViolationStashPresent      ViolationCode = "stash_present"
	ViolationReflogPresent     ViolationCode = "reflog_present"
	ViolationStateFilePresent  ViolationCode = "git_state_file_present"
	ViolationHeadMismatch      ViolationCode = "head_commit_mismatch"
	ViolationFutureCommit      ViolationCode = "future_commit_present"
	ViolationUnreachableCommit ViolationCode = "unreachable_commit_present"
	ViolationLeakArtifact      ViolationCode = "leak_artifact_present"
	ViolationHarnessTampered   ViolationCode = "harness_tampered"
)

// LeakageViolation describes a specific anti-leakage or anti-tamper finding.
type LeakageViolation struct {
	Code   ViolationCode `json:"code"`
	Target string        `json:"target,omitempty"`
	Detail string        `json:"detail"`
}

// SanitizeConfig defines what git state and workspace artifacts must be stripped or verified.
type SanitizeConfig struct {
	BaseCommit            string            `json:"base_commit,omitempty"`
	CutoffTimestamp       time.Time         `json:"cutoff_timestamp,omitempty"`
	AllowedRemotes        []string          `json:"allowed_remotes,omitempty"`
	ForbiddenCommits      []string          `json:"forbidden_commits,omitempty"`
	ForbiddenPaths        []string          `json:"forbidden_paths,omitempty"`
	ExpectedHarnessHashes map[string]string `json:"expected_harness_hashes,omitempty"`
	PruneUnreachable      bool              `json:"prune_unreachable"`
}

// DefaultSanitizeConfig returns a strict anti-leakage configuration for a base commit.
func DefaultSanitizeConfig(baseCommit string, cutoff time.Time) SanitizeConfig {
	return SanitizeConfig{
		BaseCommit:       strings.TrimSpace(baseCommit),
		CutoffTimestamp:  cutoff,
		PruneUnreachable: true,
		ForbiddenPaths: []string{
			"gold_patch.diff",
			"gold_patch.patch",
			"test_patch.diff",
			"test_patch.patch",
			".swe_bench_gold.json",
		},
	}
}

// WorkspaceAuditReport records stripped artifacts and clean-state audit results.
type WorkspaceAuditReport struct {
	WorkspacePath   string             `json:"workspace_path"`
	Clean           bool               `json:"clean"`
	HeadCommit      string             `json:"head_commit"`
	CommitCount     int                `json:"commit_count"`
	StrippedRemotes []string           `json:"stripped_remotes,omitempty"`
	StrippedTags    []string           `json:"stripped_tags,omitempty"`
	StrippedRefs    []string           `json:"stripped_refs,omitempty"`
	RemovedPaths    []string           `json:"removed_paths,omitempty"`
	Violations      []LeakageViolation `json:"violations,omitempty"`
	AuditedAt       time.Time          `json:"audited_at"`
}

// SanitizeGitWorkspace strips remotes, tags, non-HEAD branches, reflogs, stashes,
// unreachable future commits, and forbidden fix artifacts, then verifies clean state.
func SanitizeGitWorkspace(ctx context.Context, repoDir string, cfg SanitizeConfig) (*WorkspaceAuditReport, error) {
	if err := validateRepoDir(repoDir); err != nil {
		return nil, err
	}

	strippedRemotes, err := stripGitRemotes(ctx, repoDir, cfg.AllowedRemotes)
	if err != nil {
		return nil, err
	}

	if base := strings.TrimSpace(cfg.BaseCommit); base != "" {
		if _, err := runGit(ctx, repoDir, "checkout", "--force", "--detach", base); err != nil {
			return nil, fmt.Errorf("checkout base commit %s: %w", base, err)
		}
	} else {
		headSHA, err := runGit(ctx, repoDir, "rev-parse", "HEAD")
		if err != nil {
			return nil, fmt.Errorf("resolve HEAD: %w", err)
		}
		if _, err := runGit(ctx, repoDir, "checkout", "--force", "--detach", strings.TrimSpace(headSHA)); err != nil {
			return nil, fmt.Errorf("detach HEAD: %w", err)
		}
	}

	if _, err := runGit(ctx, repoDir, "reset", "--hard", "HEAD"); err != nil {
		return nil, fmt.Errorf("reset worktree: %w", err)
	}

	strippedTags, err := stripGitTags(ctx, repoDir)
	if err != nil {
		return nil, err
	}

	strippedRefs, err := stripNonHeadRefs(ctx, repoDir)
	if err != nil {
		return nil, err
	}

	removedPaths, err := removeGitMetadataAndForbiddenFiles(ctx, repoDir, cfg)
	if err != nil {
		return nil, err
	}

	if cfg.PruneUnreachable {
		if _, err := runGit(ctx, repoDir, "reflog", "expire", "--expire=now", "--all"); err != nil {
			return nil, fmt.Errorf("expire reflogs: %w", err)
		}
		if _, err := runGit(ctx, repoDir, "prune", "--expire=now"); err != nil {
			return nil, fmt.Errorf("prune unreachable objects: %w", err)
		}
		if _, err := runGit(ctx, repoDir, "gc", "--prune=now", "--quiet"); err != nil {
			return nil, fmt.Errorf("gc prune=now: %w", err)
		}
	}

	report, err := VerifyWorkspaceClean(ctx, repoDir, cfg)
	if report != nil {
		report.StrippedRemotes = strippedRemotes
		report.StrippedTags = strippedTags
		report.StrippedRefs = strippedRefs
		report.RemovedPaths = removedPaths
	}
	return report, err
}

// VerifyWorkspaceClean audits a git workspace for leaked remotes, tags, branches,
// reflogs, future/unreachable commits, forbidden fix artifacts, and harness tampering.
func VerifyWorkspaceClean(ctx context.Context, repoDir string, cfg SanitizeConfig) (*WorkspaceAuditReport, error) {
	if err := validateRepoDir(repoDir); err != nil {
		return nil, err
	}

	report := &WorkspaceAuditReport{
		WorkspacePath: repoDir,
		AuditedAt:     time.Now().UTC(),
	}

	headOut, err := runGit(ctx, repoDir, "rev-parse", "HEAD")
	if err != nil {
		return nil, fmt.Errorf("resolve HEAD: %w", err)
	}
	report.HeadCommit = strings.ToLower(strings.TrimSpace(headOut))

	if base := strings.ToLower(strings.TrimSpace(cfg.BaseCommit)); base != "" {
		if !commitMatches(report.HeadCommit, base) {
			report.Violations = append(report.Violations, LeakageViolation{
				Code:   ViolationHeadMismatch,
				Target: "HEAD",
				Detail: fmt.Sprintf("HEAD %s does not match expected base commit %s", report.HeadCommit, base),
			})
		}
	}

	allowedRemoteSet := toLowerSet(cfg.AllowedRemotes)
	remotesOut, err := runGit(ctx, repoDir, "remote")
	if err != nil {
		return nil, fmt.Errorf("list git remotes: %w", err)
	}
	for _, r := range splitNonEmptyLines(remotesOut) {
		if _, ok := allowedRemoteSet[strings.ToLower(r)]; !ok {
			report.Violations = append(report.Violations, LeakageViolation{
				Code:   ViolationRemotePresent,
				Target: r,
				Detail: fmt.Sprintf("git remote %q remains configured", r),
			})
		}
	}

	tagsOut, err := runGit(ctx, repoDir, "tag", "-l")
	if err != nil {
		return nil, fmt.Errorf("list git tags: %w", err)
	}
	for _, tag := range splitNonEmptyLines(tagsOut) {
		report.Violations = append(report.Violations, LeakageViolation{
			Code:   ViolationTagPresent,
			Target: tag,
			Detail: fmt.Sprintf("git tag %q remains in workspace", tag),
		})
	}

	refsOut, err := runGit(ctx, repoDir, "for-each-ref", "--format=%(refname)")
	if err != nil {
		return nil, fmt.Errorf("list git refs: %w", err)
	}
	for _, ref := range splitNonEmptyLines(refsOut) {
		switch {
		case strings.HasPrefix(ref, "refs/tags/"):
			report.Violations = append(report.Violations, LeakageViolation{
				Code:   ViolationTagPresent,
				Target: ref,
				Detail: fmt.Sprintf("tag ref %q still exists", ref),
			})
		case strings.HasPrefix(ref, "refs/remotes/"):
			report.Violations = append(report.Violations, LeakageViolation{
				Code:   ViolationRemotePresent,
				Target: ref,
				Detail: fmt.Sprintf("remote ref %q still exists", ref),
			})
		case ref == "refs/stash":
			report.Violations = append(report.Violations, LeakageViolation{
				Code:   ViolationStashPresent,
				Target: ref,
				Detail: "git stash ref still exists",
			})
		default:
			report.Violations = append(report.Violations, LeakageViolation{
				Code:   ViolationExtraBranch,
				Target: ref,
				Detail: fmt.Sprintf("non-detached ref %q remains reachable", ref),
			})
		}
	}

	checkGitArtifactPresence(repoDir, report)
	if err := checkCommitHistoryAndObjects(ctx, repoDir, cfg, report); err != nil {
		return nil, err
	}
	if err := checkForbiddenWorkspacePaths(repoDir, cfg, report); err != nil {
		return nil, err
	}
	if err := checkHarnessIntegrity(repoDir, cfg.ExpectedHarnessHashes, report); err != nil {
		return nil, err
	}

	report.Clean = len(report.Violations) == 0
	if !report.Clean {
		return report, fmt.Errorf("%w: %d violation(s)", ErrLeakageDetected, len(report.Violations))
	}
	return report, nil
}

func validateRepoDir(repoDir string) error {
	if strings.TrimSpace(repoDir) == "" {
		return fmt.Errorf("%w: empty workspace path", ErrLeakageDetected)
	}
	gitDir := filepath.Join(repoDir, ".git")
	info, err := os.Stat(gitDir)
	if err != nil {
		return fmt.Errorf("inspect workspace .git directory %q: %w", gitDir, err)
	}
	if !info.IsDir() {
		return fmt.Errorf("%w: %s is not a git directory", ErrLeakageDetected, gitDir)
	}
	return nil
}

func stripGitRemotes(ctx context.Context, repoDir string, allowed []string) ([]string, error) {
	allowedSet := toLowerSet(allowed)
	out, err := runGit(ctx, repoDir, "remote")
	if err != nil {
		return nil, fmt.Errorf("list remotes: %w", err)
	}
	var stripped []string
	for _, r := range splitNonEmptyLines(out) {
		if _, ok := allowedSet[strings.ToLower(r)]; ok {
			continue
		}
		if _, err := runGit(ctx, repoDir, "remote", "remove", r); err != nil {
			return nil, fmt.Errorf("remove remote %q: %w", r, err)
		}
		stripped = append(stripped, r)
	}
	sort.Strings(stripped)
	return stripped, nil
}

func stripGitTags(ctx context.Context, repoDir string) ([]string, error) {
	out, err := runGit(ctx, repoDir, "tag", "-l")
	if err != nil {
		return nil, fmt.Errorf("list tags: %w", err)
	}
	tags := splitNonEmptyLines(out)
	if len(tags) > 0 {
		args := append([]string{"tag", "-d"}, tags...)
		if _, err := runGit(ctx, repoDir, args...); err != nil {
			return nil, fmt.Errorf("delete tags: %w", err)
		}
	}
	sort.Strings(tags)
	return tags, nil
}

func stripNonHeadRefs(ctx context.Context, repoDir string) ([]string, error) {
	out, err := runGit(ctx, repoDir, "for-each-ref", "--format=%(refname)")
	if err != nil {
		return nil, fmt.Errorf("enumerate refs: %w", err)
	}
	refs := splitNonEmptyLines(out)
	for _, ref := range refs {
		if _, err := runGit(ctx, repoDir, "update-ref", "-d", ref); err != nil {
			return nil, fmt.Errorf("delete ref %q: %w", ref, err)
		}
	}
	sort.Strings(refs)
	return refs, nil
}

func removeGitMetadataAndForbiddenFiles(ctx context.Context, repoDir string, cfg SanitizeConfig) ([]string, error) {
	_ = ctx
	var removed []string

	gitCleanupPaths := []string{
		filepath.Join(".git", "logs"),
		filepath.Join(".git", "FETCH_HEAD"),
		filepath.Join(".git", "ORIG_HEAD"),
		filepath.Join(".git", "MERGE_HEAD"),
		filepath.Join(".git", "CHERRY_PICK_HEAD"),
		filepath.Join(".git", "packed-refs"),
		filepath.Join(".git", "refs", "remotes"),
		filepath.Join(".git", "refs", "tags"),
		filepath.Join(".git", "refs", "original"),
		filepath.Join(".git", "refs", "notes"),
		filepath.Join(".git", "refs", "stash"),
	}
	for _, rel := range gitCleanupPaths {
		abs := filepath.Join(repoDir, rel)
		if _, err := os.Stat(abs); err == nil {
			if err := os.RemoveAll(abs); err != nil {
				return nil, fmt.Errorf("remove %s: %w", rel, err)
			}
			removed = append(removed, rel)
		}
	}

	for _, rel := range defaultAndCustomForbiddenPaths(cfg.ForbiddenPaths) {
		abs, ok := safeJoinUnderRoot(repoDir, rel)
		if !ok {
			continue
		}
		if _, err := os.Stat(abs); err == nil {
			if err := os.RemoveAll(abs); err != nil {
				return nil, fmt.Errorf("remove forbidden artifact %s: %w", rel, err)
			}
			removed = append(removed, rel)
		}
	}

	sort.Strings(removed)
	return removed, nil
}

func checkGitArtifactPresence(repoDir string, report *WorkspaceAuditReport) {
	logsDir := filepath.Join(repoDir, ".git", "logs")
	if info, err := os.Stat(logsDir); err == nil && info.IsDir() {
		entries, err := os.ReadDir(logsDir)
		if err == nil && len(entries) > 0 {
			report.Violations = append(report.Violations, LeakageViolation{
				Code:   ViolationReflogPresent,
				Target: ".git/logs",
				Detail: "reflog directory contains history entries",
			})
		}
	}

	stateFiles := []string{
		filepath.Join(".git", "FETCH_HEAD"),
		filepath.Join(".git", "ORIG_HEAD"),
		filepath.Join(".git", "MERGE_HEAD"),
		filepath.Join(".git", "CHERRY_PICK_HEAD"),
	}
	for _, rel := range stateFiles {
		if _, err := os.Stat(filepath.Join(repoDir, rel)); err == nil {
			report.Violations = append(report.Violations, LeakageViolation{
				Code:   ViolationStateFilePresent,
				Target: rel,
				Detail: fmt.Sprintf("git state file %s may leak branch/merge history", rel),
			})
		}
	}
}

func checkCommitHistoryAndObjects(ctx context.Context, repoDir string, cfg SanitizeConfig, report *WorkspaceAuditReport) error {
	logOut, err := runGit(ctx, repoDir, "rev-list", "--all", "--format=%H:%ct")
	if err != nil {
		return fmt.Errorf("inspect git rev-list: %w", err)
	}

	count := 0
	for _, line := range splitNonEmptyLines(logOut) {
		if strings.HasPrefix(line, "commit ") {
			continue
		}
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}
		count++
		sha := strings.TrimSpace(parts[0])
		tsSec, err := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
		if err != nil {
			continue
		}
		commitTime := time.Unix(tsSec, 0).UTC()
		if !cfg.CutoffTimestamp.IsZero() && commitTime.After(cfg.CutoffTimestamp.UTC()) {
			report.Violations = append(report.Violations, LeakageViolation{
				Code:   ViolationFutureCommit,
				Target: sha,
				Detail: fmt.Sprintf("commit %s timestamp %s exceeds cutoff %s", sha, commitTime.Format(time.RFC3339), cfg.CutoffTimestamp.UTC().Format(time.RFC3339)),
			})
		}
	}
	report.CommitCount = count

	for _, forbidden := range cfg.ForbiddenCommits {
		sha := strings.TrimSpace(forbidden)
		if sha == "" {
			continue
		}
		if _, err := runGit(ctx, repoDir, "cat-file", "-e", sha+"^{commit}"); err == nil {
			report.Violations = append(report.Violations, LeakageViolation{
				Code:   ViolationFutureCommit,
				Target: sha,
				Detail: fmt.Sprintf("forbidden commit %s is still present in git object store", sha),
			})
		}
	}

	if cfg.PruneUnreachable {
		fsckOut, err := runGit(ctx, repoDir, "fsck", "--unreachable", "--no-progress")
		if err != nil {
			return fmt.Errorf("git fsck unreachable: %w", err)
		}
		for _, line := range splitNonEmptyLines(fsckOut) {
			if strings.Contains(line, "unreachable commit ") {
				report.Violations = append(report.Violations, LeakageViolation{
					Code:   ViolationUnreachableCommit,
					Target: strings.TrimSpace(line),
					Detail: fmt.Sprintf("unreachable commit remains in object database: %s", line),
				})
			}
		}
	}

	return nil
}

func checkForbiddenWorkspacePaths(repoDir string, cfg SanitizeConfig, report *WorkspaceAuditReport) error {
	for _, rel := range defaultAndCustomForbiddenPaths(cfg.ForbiddenPaths) {
		abs, ok := safeJoinUnderRoot(repoDir, rel)
		if !ok {
			return fmt.Errorf("%w: forbidden path %q escapes workspace root", ErrLeakageDetected, rel)
		}
		if _, err := os.Stat(abs); err == nil {
			report.Violations = append(report.Violations, LeakageViolation{
				Code:   ViolationLeakArtifact,
				Target: rel,
				Detail: fmt.Sprintf("forbidden patch/leak artifact %q exists in workspace", rel),
			})
		}
	}
	return nil
}

func checkHarnessIntegrity(repoDir string, expected map[string]string, report *WorkspaceAuditReport) error {
	if len(expected) == 0 {
		return nil
	}
	keys := make([]string, 0, len(expected))
	for k := range expected {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, rel := range keys {
		want := strings.ToLower(strings.TrimSpace(expected[rel]))
		abs, ok := safeJoinUnderRoot(repoDir, rel)
		if !ok {
			report.Violations = append(report.Violations, LeakageViolation{
				Code:   ViolationHarnessTampered,
				Target: rel,
				Detail: fmt.Sprintf("harness path %q escapes workspace root", rel),
			})
			continue
		}
		got, err := fileSHA256(abs)
		if err != nil {
			report.Violations = append(report.Violations, LeakageViolation{
				Code:   ViolationHarnessTampered,
				Target: rel,
				Detail: fmt.Sprintf("protected harness file %q missing or unreadable: %v", rel, err),
			})
			continue
		}
		if got != want {
			report.Violations = append(report.Violations, LeakageViolation{
				Code:   ViolationHarnessTampered,
				Target: rel,
				Detail: fmt.Sprintf("harness file %q digest %s != expected %s", rel, got, want),
			})
		}
	}
	return nil
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func safeJoinUnderRoot(root, rel string) (string, bool) {
	cleanRoot := filepath.Clean(root)
	cleanRel := filepath.Clean(rel)
	if filepath.IsAbs(cleanRel) || cleanRel == ".." || strings.HasPrefix(cleanRel, ".."+string(filepath.Separator)) {
		return "", false
	}
	joined := filepath.Join(cleanRoot, cleanRel)
	relCheck, err := filepath.Rel(cleanRoot, joined)
	if err != nil || relCheck == ".." || strings.HasPrefix(relCheck, ".."+string(filepath.Separator)) {
		return "", false
	}
	return joined, true
}

func defaultAndCustomForbiddenPaths(custom []string) []string {
	defaults := []string{
		"gold_patch.diff",
		"gold_patch.patch",
		"test_patch.diff",
		"test_patch.patch",
		".swe_bench_gold.json",
	}
	seen := make(map[string]struct{}, len(defaults)+len(custom))
	var merged []string
	for _, p := range append(defaults, custom...) {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if _, ok := seen[p]; !ok {
			seen[p] = struct{}{}
			merged = append(merged, p)
		}
	}
	sort.Strings(merged)
	return merged
}

func commitMatches(full, candidate string) bool {
	if len(candidate) < 7 {
		return false
	}
	return full == candidate || strings.HasPrefix(full, candidate)
}

func splitNonEmptyLines(s string) []string {
	var out []string
	for _, line := range strings.Split(s, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

func runGit(ctx context.Context, repoDir string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = repoDir
	cmd.Env = append(os.Environ(),
		"GIT_CONFIG_GLOBAL=/dev/null",
		"GIT_CONFIG_NOSYSTEM=1",
		"GIT_TERMINAL_PROMPT=0",
	)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return stdout.String(), fmt.Errorf("git %s: %w (%s)", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return stdout.String(), nil
}
