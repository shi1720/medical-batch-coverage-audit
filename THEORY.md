# Coverage statement and implementation mapping

Let the training data, validation-selected weights, preprocessing, and finite companion menu be fixed independently of calibration/evaluation examples. For each context j, the query's score is s_j(x,y)=1-p_j(y|x). Each query is evaluated in a separate group containing that query and the fixed companions. Learned parameters and running normalization state are frozen. Therefore the complete context-score vector is a deterministic function of one example, even though the underlying normalization uses a batch.

For calibration size n and error level alpha, k=ceil((n+1)(1-alpha)). A finite scalar quantile is the kth ascending calibration score; use infinity when k>n. Including ties with <= is conservative.

## Maximum-score calibration (existing robust split conformal)

Define S(x,y)=max_j s_j(x,y). Exchangeability of n calibration examples and one future example implies exchangeability of their scalar S scores. The rank argument gives P(S(X,Y)<=q)>=1-alpha for q defined with the corrected split-conformal order statistic and infinity convention. Since S(X,Y)<=q is exactly the event s_j(X,Y)<=q for every j, the result is simultaneous true-label inclusion over the fixed menu. No independence among context scores is required. This is an application of standard split conformal and robust conformal calibration, not a new general theorem.

This probability is marginal over calibration and the query, conditional on fixed training/model/menu. It does not guarantee at least 90% coverage for every realized calibration split, every class, or every individual query. Random permutation of a fixed held-out image pool supports a finite-population image-level analysis; it cannot establish transportability to future patients or clinical populations.

A realized scheduler may choose any context in the protected menu, even using the query, without weakening the all-context event. A new context outside that menu is not covered by the statement. Any selection of the protected menu based on calibration/evaluation outcomes needs a separate validity argument; the present menus use only training data and independent random seeds.

The deployed set C_j requires one context pass after thresholds are fitted. The intersection over all context-specific sets is a companion-independent output but requires M query-context passes. These are different computational products. Mean deployed set size and intersection size are both recorded; the paper's primary efficiency metric is the former.

## Context-specific calibration

Calibrating each fixed context at alpha gives each context a marginal coverage guarantee. It does not imply a 1-alpha guarantee for their intersection. The elementary union bound only gives max(0,1-M*alpha). Failures in different contexts can occur on different queries. The empirical mean-minus-simultaneous gap measures that operational distinction. This is not a new multiple-testing theorem.

## Bonferroni

Use alpha/M for every context. A union bound yields the simultaneous 1-alpha lower bound under the same fixed-score exchangeability assumption. If the requested order statistic exceeds n, infinity is necessary; substituting the maximum observed score would lose the advertised finite-sample guarantee. For BreastMNIST n=77, M=12, alpha=.1, ceil(78*(1-.1/12))=78>77, so each threshold is infinity and both labels are always included.

## Corrected Max-Rank (Schlembach, Smirnov & Winands, 2026)

Each context is treated as one score dimension. The target vector is (Y,...,Y), so context dimensions are dependent, which is allowed. Each score function depends on its own proposed label coordinate and the same query x; the score vector remains examplewise and dimension-separable. Rank calibration scores within each context, using the maximum rank for ties; calculate a corrected scalar quantile of the maximum ranks over contexts. If that rank threshold is r, use the (r+1)st ascending calibration score in each context (zero-based array index r), with infinity if unavailable. The finite-sample theorem is from the cited paper; it is not newly proved here. The implementation uses the corrected score-space mapping rather than the original Max-Rank mapping. A one-context example and leave-one-out finite-pool tests exercise the extra order statistic and ties.

## Scope exclusions

No claim is made for gradient-updated/continual model states, calibration samples sharing one adaptive BN batch with one another, arbitrary unseen companion pools, external shift, rare-class conditional coverage, simultaneous inclusion across different patients, or patient-level independence absent suitable identifiers/sampling. The held-out-menu CSV is an empirical sensitivity diagnostic only. Unit tests and numerical replay verify implementation properties; they do not constitute clinical validation or independent confirmation of novelty.

References: Angelopoulos & Bates (2023), DOI 10.1561/2200000101; Massena et al. (ICML 2025), https://proceedings.mlr.press/v267/massena25a.html; Timans et al. (AISTATS 2025), https://proceedings.mlr.press/v258/timans25a.html; Schlembach et al. (COPA 2026), https://proceedings.mlr.press/v329/schlembach26a.html.

### Ties in the CMR implementation

The cited CMR equivalence theorem states an almost-sure distinct-score assumption. The present floating-point implementation instead uses maximum ranks for ties and closed score thresholds. A conservative-extension argument is needed rather than silently assuming distinctness. Add independent, bounded continuous positive jitter of amplitude delta to calibration and future-query scores. For sufficiently small delta, distinct original calibration-score groups retain their order; every jittered within-group rank is at most that example's original maximum-tie rank. Thus the original maximum-rank quantile is at least the jittered quantile. Every jittered per-context threshold is at most the corresponding original closed threshold plus delta. The no-ties CMR theorem applies to the jittered scores. Taking delta down to zero yields the closed-threshold coverage lower bound for the maximum-tie implementation (finite-score, fixed-menu setting); cases with infinite thresholds are immediate. This is a conservative implementation argument, not a claim of a new CMR method. The leave-one-out tests include quantized ties as an additional finite computational check; those tests alone are not the proof.
