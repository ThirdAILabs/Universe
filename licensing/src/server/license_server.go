package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
)

func init() {
	globalTracker.Lock()
	globalTracker.resetHeartbeatTracker()
	globalTracker.Unlock()
	MaxActiveMachines = int(parseFromVar(MaxActiveMachinesString))
	ActiveTimeoutMillis = parseFromVar(ActiveTimeoutMillisString)
}

func parseFromVar(varToParse string) int64 {
	// See https://stackoverflow.com/q/21532113/golang-string-to-int64
	parsed, err := strconv.ParseInt(varToParse, 10, 64)
	if err != nil {
		panic(err)
	}
	return parsed
}

// We do not allow more than this many machines active at one time
// We have a string variable and an int variable because we can only set strings
// during the go build phase.
var MaxActiveMachinesString = "5"
var MaxActiveMachines int = 5

// Machines we have not heard from in this many milliseconds we consider no
// longer active and do not count them towards a groups' machine limit.
// We have a string variable and an int variable because we can only set strings
// during the go build phase.
var ActiveTimeoutMillisString = "1000000"
var ActiveTimeoutMillis int64 = 1000000

// The single global machine tracker for the server
var globalTracker MachineTracker

type VerifyRequest struct {
	// Machine id to track currently licensed machines
	MachineId string `json:"machine_id"`
	// Variable metadata that we can sign to ensure the response came from us
	Metadata string `json:"metadata"`
}

func heartbeat(response_writer http.ResponseWriter, req *http.Request) {
	if req.Method != "POST" {
		http.Error(response_writer, "Invalid request method.", 405)
		return
	}

	decoder := json.NewDecoder(req.Body)
	decoder.DisallowUnknownFields() // catch unwanted fields

	var parsedReq VerifyRequest
	err := decoder.Decode(&parsedReq)
	if err != nil {
		http.Error(response_writer, fmt.Sprintf("Error parsing body: %s", err), 400)
		return
	}

	globalTracker.Lock()
	success := globalTracker.tryAddingNewMachine(parsedReq.MachineId)
	globalTracker.Unlock()

	if !success {
		http.Error(response_writer, "Every machine slot is currently taken", 400)
		return
	}

	toSign := fmt.Sprintf("%s\n%s", parsedReq.MachineId, parsedReq.Metadata)
	signatureBytes := []byte(Sign(toSign))
	response_writer.Write(signatureBytes)
}

func main() {
	http.HandleFunc("/heartbeat", heartbeat)
	log.Fatal(http.ListenAndServe(":8080", nil))
}
