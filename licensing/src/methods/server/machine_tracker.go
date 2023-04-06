package main

import (
	"sync"
	"time"
)

// The machine tracker's lock should be called before all instance methods
type MachineTracker struct {
	sync.Mutex
	machineIdToLastHeartbeat map[string]int64
}

// The machine tracker's lock should be acquired before this function
func (tracker *MachineTracker) resetHeartbeatTracker() {
	tracker.machineIdToLastHeartbeat = make(map[string]int64)
}

// The machine tracker's lock should be acquired before this function
func (tracker *MachineTracker) getNumActiveMachines(currentTime int64) int {
	numActiveMachines := 0
	for _, lastHeartbeat := range tracker.machineIdToLastHeartbeat {
		timeDifferenceMillis := currentTime - lastHeartbeat
		if timeDifferenceMillis < ActiveTimeoutMillis {
			numActiveMachines++
		}
	}
	return numActiveMachines
}

// The machine tracker's lock should be acquired before this function
func (tracker *MachineTracker) tryAddingNewMachine(machineId string) bool {
	currentTime := time.Now().UnixMilli()
	numActiveMachines := tracker.getNumActiveMachines(currentTime)

	numMachinesThatWillBeAdded := 0
	_, keyExists := tracker.machineIdToLastHeartbeat[machineId]
	if !keyExists {
		numMachinesThatWillBeAdded++
	}

	if numActiveMachines+numMachinesThatWillBeAdded > MaxActiveMachines {
		return false
	}

	tracker.machineIdToLastHeartbeat[machineId] = currentTime

	return true
}
