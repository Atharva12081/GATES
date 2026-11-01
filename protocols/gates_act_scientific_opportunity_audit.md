# GATES-ACT scientific-opportunity audit protocol

Status: frozen before inspecting the row-level outcomes of the two known missed development run
groups. The aggregate fact that the highest-three-dose policy missed two of 83 active run groups
was already known. This audit is diagnostic development analysis, not independent confirmation and
not a threshold-selection exercise.

## Population and evidence boundary

- Use the 106 fully observed development run groups (99 compound groups) already represented in
  `direct_discovery_decisions.parquet`.
- Use the existing out-of-fold fold assignment and saved predictions. Do not refit on held-out
  outcomes for the identifiability analysis.
- A treated response cell is an effect when its absolute normalized response is at least 0.40.
- The physical budget is four day-matched vehicles plus three treated doses on each of four days:
  16/32 cells, with 12/28 treated response cells entering decision metrics.

## Prespecified biological patterns

- **High-dose missed active run group:** at least one reference effect exists, but no rank-5, 6 or
  7 measurement is an effect.
- **Lower-dose-only:** at least one effect occurs at ranks 1-4 and no effect occurs at ranks 5-7.
- **Transient:** at least one effect occurs on day indices 0-2 and none occurs on day index 3.
- **Delayed:** no effect occurs on day indices 0-1 and at least one occurs on day indices 2-3.
- These categories may overlap. Report their cell locations, magnitudes and source; do not infer a
  biological mechanism from response shape alone.

## Legal identifiability analysis

Restrict to run groups with no directly observed high-dose effect. Evaluate two scores derived only
from legal observations or saved group-disjoint predictions:

1. maximum absolute observed rank-5-to-7 response;
2. maximum same-sign model-agreement score over hidden cells, defined as the smaller absolute
   prediction when GATES and the residual model agree in sign, otherwise zero.

Report the rank, AUROC and average precision for separating hidden-active from inactive run groups.
Because only two hidden-active cases are expected, these are descriptive diagnostics with exact
counts, not stable performance estimates. Do not tune a threshold.

## Oracle upper bounds

- **Run-group discovery oracle:** a truth-informed design can select one effect cell whenever an
  active run group contains any effect. Its ceiling is 100% active-group sensitivity. Report the
  absolute number and percentage-point opportunity above fixed high dose.
- **Direct cell-discovery oracle:** independently on each assay day, select up to three effect cells
  among seven treated doses. Sum `min(3, number of effect cells that day)` and divide by all effect
  cells. This is an unattainable diagnostic ceiling, not an implementable acquisition policy.
- Report the oracle gain above fixed-high direct capture and above full fixed-high GATES recall.

## Go/no-go rule for one final algorithm

Pursue a new method only if all conditions hold:

1. fixed high dose misses at least 5% of active run groups or at least five active run groups;
2. the run-group oracle offers at least five percentage points of additional sensitivity;
3. at least one legal score achieves leave-group-out/saved-OOF AUROC at least 0.80 and average
   precision at least 0.30 among high-dose-negative groups;
4. the opportunity is not solely additional response-cell mapping in run groups already known to
   be active.

Otherwise stop algorithmic searching and preserve the negative evidence.
