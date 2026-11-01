# GATES-ACT v3 dataset compatibility record

| Candidate | Dose × time grid | Neural electrophysiology | Matched controls | Group IDs | Decision |
|---|---|---|---|---|---|
| NTP/EPA Phase-1 DNT MEA | Yes | Mean firing rate | Yes | DTXSID | Include |
| Brown 2016 MEA | Yes, six treated ranks | Mean firing rate | Yes | Compound and plate | Include as external confirmation |
| EPA Tipping Point | Yes | Mean firing rate | Yes | Compound and plate | Include as final confirmation |
| G-Node hPSC/rat raw MEA | Development and pharmacology recordings, 2.3 TiB | Raw voltage | Study-specific | Culture metadata | Reject for this benchmark: incompatible scale and acquisition unit |
| DNT-DIVER non-MEA assays | Mostly concentration-response summaries | Mixed morphology/viability | Assay-specific | Chemical | Reject pooling: different endpoint and no repeated MEA schedule |
| Public organ-on-chip candidates found | Terminal omics, viability, morphology, or proprietary examples | No compatible repeated neural MEA endpoint | Mixed | Mixed | Reject pooling: estimand and laboratory cost are not commensurate |

The compatibility rule was established before quantitative pooling: a dataset must expose
well-level neural electrophysiology, repeated chronological measurements, dose identity, matched
control context, and a defensible measurement-cost unit. Rejected sources may support future
transfer studies but cannot enlarge the present sample size.

