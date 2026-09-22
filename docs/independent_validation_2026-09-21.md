# Independent synthetic validation — frozen protocol

Before viewing the new results, fix the following continuation of the initial
benchmark. Train on two fresh data realizations (seeds 873 and 874), with two
training seeds (2025, 2026), on the unresolved redundant graph. Connectivity and
all model hyperparameters remain unchanged. These are independent noisy time
series, not independent graph topologies. No IBL data are accessed.

For each latent, compute attribution rankings from validation inputs. Freeze the
encoder and the validation-selected Ridge decoder. On held-out test inputs,
replace each selected neuron's entire temporal window with its training mean.
Compare R² drops for the top six neurons, bottom six, the true connected six,
the disconnected six, and 99 fixed random groups of six. Positive drops indicate
harm to decoding; negative drops are retained. No model or decoder is refitted.
The unperturbed test score must equal the ordinary decoder test score.

Report every result, support AUROC, seed stability, and the paired differences
between top-group and bottom/random-group drops. Random groups are a descriptive
reference, not a permutation significance test. Training-mean masking can create
out-of-distribution inputs; it measures model reliance rather than biological
causality. Ground-truth connected/disconnected masks are an independent sanity
check and must not determine attribution rankings.

Decision rule: do not call attribution consistently supported if either
top-versus-bottom or top-versus-random-median direction fails in any of the eight
latent/data-seed/training-seed cases per family. Passing would justify further
limited validation, not full-cohort or causal claims. This conservative check is
not a formal statistical acceptance test. Do not tune this rule or model settings
after examining the new test results.

Remaining external requirements: verified animal identifiers for animal-level
inference, and the authors' matching results/protocol for direct published-RRR
comparison. Local RRR/Ridge comparisons do not require the authors' response.

## Results and deadline decision

Both new data realizations completed. Across the eight variable/data/training-seed
cases per family, the existing adaptation had mean support AUROC 0.688 (range
0.222–0.833), mean test decoding R² 0.812, and its top-ranked neuron group caused
a larger held-out R² drop than the bottom and random-median group in 7/8 cases.
The true connected group harmed decoding more than the disconnected group in 8/8.

The multiobjective prototype had mean support AUROC 0.774 (0.417–0.944), mean test
R² 0.855, and passed both ranking-direction checks in 8/8 cases. This supports
continued investigation but does not justify replacing the production pipeline
immediately: it has not been integrated with IBL trial boundaries, categorical
variables, portable jobs, recovery, nulls, or the analysis pipeline. Implementing
and revalidating all of that before training would increase schedule risk.

For the 8 October report deadline, proceed with a frozen 48-session exploratory
study using the tested regularized CEBRA adaptation. Treat held-out decoding and
comparison with identical-input local linear baselines as primary. Treat neuron
and area attribution as exploratory model sensitivity, report seed stability and
observed/null results, and do not call it canonical xCEBRA, causal identification,
or a direct replication of the published RRR paper. The confirmatory 41-session
reserve stays untouched.

The run uses the first 48 exploratory sessions in the pre-existing cohort order,
three training seeds, fixed dimension 4, 1,000 steps, one observed and one
trial-shuffled condition, shared attribution samples, 500 bootstrap draws, and
two independent P6000 workers. This is at most 2,304 encoder fits. The measured
69.1-minute one-session calibration estimates about 27.6 hours at ideal two-GPU
utilization; allow 28–40 hours. One calibration occupied 590 MB, projecting about
28 GB; Loki had 3.0 TB free at preflight.
