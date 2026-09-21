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
