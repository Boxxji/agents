//go:build !race

package longmemeval

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/hanzoai/agents/control-plane/internal/storage"
	"github.com/stretchr/testify/require"
)

// TestRealLocalStorageParity verifies exact behavioral parity between
// InMemoryStore and Hanzo Agents' real canonical storage.LocalStorage
// (BoltDB KV + SQLite vector/event store) across all 5 LongMemEval categories.
//
// Guarded with !race because control-plane/go.mod pins legacy
// github.com/boltdb/bolt v1.3.1, whose unsafe page slice casts trigger Go's
// checkptr detector automatically enabled by -race.
func TestRealLocalStorageParity(t *testing.T) {
	ctx := context.Background()
	tempDir := t.TempDir()

	factory := func() (MemoryStore, func(), error) {
		subDir := filepath.Join(tempDir, fmt.Sprintf("run-%d", time.Now().UnixNano()))
		ls := storage.NewLocalStorage(storage.LocalStorageConfig{})
		cfg := storage.StorageConfig{
			Mode: "local",
			Local: storage.LocalStorageConfig{
				DatabasePath: filepath.Join(subDir, "agents.db"),
				KVStorePath:  filepath.Join(subDir, "agents.bolt"),
			},
			Vector: storage.VectorStoreConfig{
				Distance: "cosine",
			},
		}
		if err := ls.Initialize(ctx, cfg); err != nil {
			return nil, nil, err
		}
		cleanup := func() {
			_ = ls.Close(ctx)
		}
		return ls, cleanup, nil
	}

	// Pick one representative case from each of the 5 LongMemEval categories.
	full := BuildDefaultSuite(DefaultSeed)
	var sample []EvalCase
	seen := make(map[TaskCategory]bool)
	for _, tc := range full {
		if !seen[tc.Category] {
			seen[tc.Category] = true
			sample = append(sample, tc)
		}
	}
	require.Len(t, sample, 5)

	// Verify LocalStorage boots cleanly (skip only if SQLite lacks FTS5).
	probeStore, probeCleanup, err := factory()
	if err != nil {
		if strings.Contains(err.Error(), "no such module: fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping LocalStorage parity test")
		}
		require.NoError(t, err)
	}
	probeCleanup()
	_ = probeStore

	policies := []MemoryPolicy{
		FlatKVLatestPolicy{},
		VectorLexicalRecencyPolicy{Embedder: NewHashEmbedder(256), TopK: 8, RecencyWeight: 0.25},
		KaiEvidenceGatedPolicy{MinConfidence: 0.55, MinEvidenceScore: 0.35, Embedder: NewHashEmbedder(256)},
	}

	memRes, err := EvaluateSuite(ctx, policies, sample, nil, true)
	require.NoError(t, err)

	localRes, err := EvaluateSuite(ctx, policies, sample, factory, true)
	require.NoError(t, err)

	require.Len(t, localRes, len(memRes))
	for i := range memRes {
		require.Equal(t, memRes[i].PolicyName, localRes[i].PolicyName)
		require.Equal(t, memRes[i].OverallAccuracy, localRes[i].OverallAccuracy, "LocalStorage parity mismatch on %s", memRes[i].PolicyName)
		require.Equal(t, memRes[i].StaleFactErrorRate, localRes[i].StaleFactErrorRate)
		require.Equal(t, memRes[i].CrossTenantLeaks, localRes[i].CrossTenantLeaks)
		require.Equal(t, 0, localRes[i].CrossTenantLeaks)
		for j := range memRes[i].Outcomes {
			require.Equal(t, memRes[i].Outcomes[j].Passed, localRes[i].Outcomes[j].Passed,
				"case %s mismatch on %s", memRes[i].Outcomes[j].CaseID, memRes[i].PolicyName)
		}
	}
}
