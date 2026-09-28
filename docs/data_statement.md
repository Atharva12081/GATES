# Data Statement

## Source

The feasibility run uses three Figure 3 CSV tables published in the public OrganoID GitHub
repository: organoid measurements, aggregate propidium iodide fluorescence measurements, and MTS
endpoint viability. `scripts/fetch_data.py` downloads exact URLs and records SHA-256 hashes.

## Units and grouping

The endpoint table contains 21 dosage–replicate units across seven gemcitabine doses and three
replicates. Each unit contributes a trajectory sampled every four hours. All time points from one
unit remain in a single evaluation partition. The experiment rotates complete replicates through
train, calibration, and test roles.

## Endpoint

The continuous endpoint is MTS viability. The predefined classification endpoint marks viability
below 0.5 as a low-viability response. The raw continuous value is retained in all evidence tables.

## Missingness and joins

The PI and organoid tables expose six replicates per dose, but the endpoint table exposes only three.
The analysis uses the 21 units with an independently measured endpoint. Image-derived morphology is
left-joined to aggregate PI trajectories at dosage, replicate, and time; missing feature values are
imputed within each training fold.

## License and redistribution

At build time, the upstream repository had no GitHub-detected license. Raw tables are therefore
downloaded locally and ignored by version control. Public redistribution and competition use should
be confirmed with the original authors.

