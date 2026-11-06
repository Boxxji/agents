package ui

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/hanzoai/agents/control-plane/internal/services"
	"github.com/hanzoai/agents/control-plane/internal/storage"
	"github.com/hanzoai/agents/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// getAgentStub stubs the one storage call that decides the status code; the
// shared MockStorageProvider is behind the integration build tag.
type getAgentStub struct {
	storage.StorageProvider
	err error
}

func (s getAgentStub) GetAgent(context.Context, string) (*types.AgentNode, error) {
	return nil, s.err
}

func TestGetNodeDetailsHandlerErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name       string
		err        error
		wantStatus int
		wantBody   string
	}{
		{"missing node", fmt.Errorf("%w: ID 'n1'", storage.ErrAgentNodeNotFound), http.StatusNotFound, "not found"},
		{"storage failure", errors.New("db password=secret unavailable"), http.StatusInternalServerError, "failed to retrieve"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := NewNodesHandler(services.NewUIService(getAgentStub{err: tt.err}, nil, nil, nil))
			router := gin.New()
			router.GET("/nodes/:nodeId/details", handler.GetNodeDetailsHandler)

			w := httptest.NewRecorder()
			router.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/nodes/n1/details", nil))

			if w.Code != tt.wantStatus {
				t.Fatalf("status = %d, want %d", w.Code, tt.wantStatus)
			}
			if !strings.Contains(w.Body.String(), tt.wantBody) {
				t.Fatalf("body = %s, want it to contain %q", w.Body, tt.wantBody)
			}
			if strings.Contains(w.Body.String(), "secret") {
				t.Fatalf("body leaks the internal error: %s", w.Body)
			}
		})
	}
}
