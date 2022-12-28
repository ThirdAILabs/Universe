package main

import (
	"bytes"
	"crypto/ed25519"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func makeHeartbeatRequest(t *testing.T, body string) (int, string) {
	bodyJsonBytes := []byte(body)
	bodyBuffer := bytes.NewBuffer(bodyJsonBytes)
	req := httptest.NewRequest(http.MethodPost, "/heartbeat", bodyBuffer)
	wr := httptest.NewRecorder()
	Heartbeat(wr, req)
	res := wr.Result()
	defer res.Body.Close()

	responseBody, err := ioutil.ReadAll(res.Body)
	assert.Equal(t, err, nil)

	return res.StatusCode, string(responseBody)
}

func TestBadBody(t *testing.T) {
	statusCode, body := makeHeartbeatRequest(t, "Not Valid Json")
	assert.Equal(t, statusCode, 400)
	assert.Regexp(t, ".*invalid character 'N' looking for beginning of value", body)
}

func TestExtraFieldsInBody(t *testing.T) {
	statusCode, body := makeHeartbeatRequest(t, "{\"test\": \"test\"}")
	assert.Equal(t, statusCode, 400)
	assert.Regexp(t, ".*unknown field \"test\"", body)
}

func TestGoodRequest(t *testing.T) {
	ResetGlobalMachineHeartbeatTracker()
	startTime := time.Now().Unix()

	statusCode, _ := makeHeartbeatRequest(t, "{\"machine_id\": \"123\", \"metadata\": \"\"}")
	assert.Equal(t, statusCode, 200)

	assert.Equal(t, globalTracker.GetNumActiveMachines(startTime), 1)
	currentTime := time.Now().Unix()
	assert.Equal(t, globalTracker.GetNumActiveMachines(currentTime+ActiveTimeoutMillis), 0)
}

func TestMultipleMachineTracking(t *testing.T) {
	ResetGlobalMachineHeartbeatTracker()
	startTime := time.Now().Unix()

	print(MaxActiveMachines, "\n")

	for machineId := 0; machineId < MaxActiveMachines; machineId++ {
		statusCode, _ := makeHeartbeatRequest(t, fmt.Sprintf("{\"machine_id\": \"%d\", \"metadata\": \"\"}", machineId))
		assert.Equal(t, statusCode, 200)
	}

	assert.Equal(t, globalTracker.GetNumActiveMachines(startTime), 5)
}

func TestTooManyMachines(t *testing.T) {
	ResetGlobalMachineHeartbeatTracker()
	startTime := time.Now().Unix()

	for machineId := 0; machineId < MaxActiveMachines+10; machineId++ {
		statusCode, body := makeHeartbeatRequest(t, fmt.Sprintf("{\"machine_id\": \"%d\", \"metadata\": \"\"}", machineId))
		if machineId >= MaxActiveMachines {
			assert.Equal(t, statusCode, 400)
			assert.Equal(t, "Every machine slot is currently taken\n", body)
		} else {
			assert.Equal(t, statusCode, 200)
		}
	}

	assert.Equal(t, globalTracker.GetNumActiveMachines(startTime), 5)
}

func TestSignature(t *testing.T) {
	ResetGlobalMachineHeartbeatTracker()
	messageBytes := []byte("123\nabc")
	_, body := makeHeartbeatRequest(t, "{\"machine_id\": \"123\", \"metadata\": \"abc\"}")
	signatureBytes, err := base64.StdEncoding.DecodeString(body)
	if err != nil {
		panic(err)
	}
	assert.True(t, ed25519.Verify(publicKey, messageBytes, signatureBytes))
}

// TODO: Add timeout test
