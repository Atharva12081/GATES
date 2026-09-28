# Five Minute Demo Script

## 0:00 to 0:30 Problem

Longitudinal experiments collect every scheduled observation even when the endpoint may already be
clear. Stopping blindly saves time but risks the wrong conclusion.

## 0:30 to 1:05 Concept

GATES watches only the measurements available so far. It returns CONTINUE while evidence is weak,
STOP when a calibration rule passes, and ABSTAIN when the trajectory is too unfamiliar for early
automation.

## 1:05 to 2:05 Held-out case

Open a held-out OrganoID unit. Move through the prediction trajectory, show the selected decision
time, endpoint confidence, observations saved, and final viability. Open the audit record.

## 2:05 to 2:45 Failure-aware case

Select an ABSTAIN or late case. Show that the feature-distance gate prevents an early automated call.

## 2:45 to 3:40 Evidence

Show performance by time and all held-out decisions. Explain that the split rotates complete
replicates through train, calibration, and test. No frame crosses partitions.

## 3:40 to 4:20 Honesty

The dataset has only 21 endpoint-labelled units. The code calculates a conservative confidence
bound and does not claim certified low risk when the calibration sample cannot support it.

## 4:20 to 5:00 Reproduction and next step

Show the one-command run, data manifest, locked environment, tests, and evidence tables. The next
gate is a larger independent longitudinal dataset that can support stronger calibration claims.

