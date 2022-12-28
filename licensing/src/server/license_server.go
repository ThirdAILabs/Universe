package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"sync"
	"time"
)

func init() {
	ResetGlobalMachineHeartbeatTracker()
	MaxActiveMachines = parseFromVar(MaxActiveMachinesString)
}

func parseFromVar(varToParse string) int {
	parsed, err := strconv.Atoi(varToParse)
	if err != nil {
		panic(err)
	}
	return parsed
}

// We do not allow more than this many machines active at one time
// We have a string variable and an int variable because we can only set strings 
// during the go build phase.
var MaxActiveMachinesString = "5"
var MaxActiveMachines = 5

const ActiveTimeoutMillis int64 = 1000000

type MachineTracker struct {
	Mu                       sync.Mutex
	MachineIdToLastHeartbeat map[string]int64
}

func (tracker *MachineTracker) GetNumActiveMachines(currentTime int64) int {
	numActiveMachines := 0
	for _, lastHeartbeat := range tracker.MachineIdToLastHeartbeat {
		timeDifferenceMillis := currentTime - lastHeartbeat
		if timeDifferenceMillis < ActiveTimeoutMillis {
			numActiveMachines++
		}
	}
	return numActiveMachines
}

var globalTracker MachineTracker

type VerifyRequest struct {
	// Machine id to track currently licensed machines
	MachineId string `json:"machine_id"`
	// Variable metadata that we can sign to ensure the response came from us
	Metadata string `json:"metadata"`
}

func ResetGlobalMachineHeartbeatTracker() {
	globalTracker.Mu.Lock()
	defer globalTracker.Mu.Unlock()
	globalTracker.MachineIdToLastHeartbeat = make(map[string]int64)
}

func Heartbeat(wr http.ResponseWriter, req *http.Request) {
	if req.Method != "POST" {
		http.Error(wr, "Invalid request method.", 405)
		print("Returning 405 bad method\n")
		return
	}

	decoder := json.NewDecoder(req.Body)
	decoder.DisallowUnknownFields() // catch unwanted fields

	var parsedReq VerifyRequest
	err := decoder.Decode(&parsedReq)
	if err != nil {
		http.Error(wr, fmt.Sprintf("Error parsing body: %s", err), 400)
		print("Returning 400 bad body\n")
		return
	}

	globalTracker.Mu.Lock()

	currentTime := time.Now().Unix()
	numActiveMachines := globalTracker.GetNumActiveMachines(currentTime)

	numMachinesThatWillBeAdded := 0
	_, keyExists := globalTracker.MachineIdToLastHeartbeat[parsedReq.MachineId]
	if !keyExists {
		numMachinesThatWillBeAdded++
	}

	if numActiveMachines+numMachinesThatWillBeAdded > MaxActiveMachines {
		http.Error(wr, "Every machine slot is currently taken", 400)
		print("Returning all machine slots currently taken\n")
		globalTracker.Mu.Unlock()
		return
	}

	globalTracker.MachineIdToLastHeartbeat[parsedReq.MachineId] = currentTime
	globalTracker.Mu.Unlock()

	toSign := fmt.Sprintf("%s\n%s", parsedReq.MachineId, parsedReq.Metadata)
	signatureBytes := []byte(Sign(toSign))
	wr.Write(signatureBytes)
	print("Returning success\n")
}

func main() {
	print(MaxActiveMachines)
	http.HandleFunc("/heartbeat", Heartbeat)
	log.Fatal(http.ListenAndServe(":8080", nil))
}
