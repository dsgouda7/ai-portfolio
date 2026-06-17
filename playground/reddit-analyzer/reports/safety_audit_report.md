# Safety Audit Report

Generated (UTC): 2026-06-16T23:23:01.645947+00:00

## Scope

- Total clustered negative comments: 37,536
- Clusters discovered: 2

## Cluster-Metadata Correlation

Spearman correlation matrix:

|                    |   cluster_id |   depth |   sentiment_compound |   hour_of_day |
|:-------------------|-------------:|--------:|---------------------:|--------------:|
| cluster_id         |    1         |     nan |             0.226409 |     0.0881006 |
| depth              |  nan         |     nan |           nan        |   nan         |
| sentiment_compound |    0.226409  |     nan |             1        |    -0.190255  |
| hour_of_day        |    0.0881006 |     nan |            -0.190255 |     1         |

Kruskal-Wallis p-values across clusters:

- Thread depth variance by cluster: nan
- Sentiment variance by cluster: 0

## Detection Drift / Coverage Gap

- Baseline lexicon size: 20
- Cluster-discovered keyword size: 394
- Overlap count: 0
- Coverage: 0.00%
- Coverage gap: 100.00%
- Detection drift: 100.00%

## Cluster Summaries

### Cluster 0 - just | like

- Comments: 35209
- Avg sentiment: -0.728
- Avg depth: 0.00
- Avg score: 164.64
- Peak hour (UTC): 22
- Risk summary: Heuristic label fallback (LLM unavailable).
- Top keywords: just, like, people, time, said, don, know, man, ve, did, world, life, way, day, death, years, going, didn, eli5, think, make, eyes, body, right, does

### Cluster 1 - er | det

- Comments: 2327
- Avg sentiment: -0.496
- Avg depth: 0.00
- Avg score: 0.00
- Peak hour (UTC): 22
- Risk summary: Heuristic label fallback (LLM unavailable).
- Top keywords: er, det, jeg, og, ikke, en, der, på, til, har, så, af, med, den, det er, man, som, du, kan, et, men, om, var, hvis, være

## Coverage Gap Highlights

- Missing baseline terms (sample): backward, criminal, degenerate, disgusting, filthy, garbage, hate, idiot, inferior, invader, parasite, savage, scum, stupid, subhuman, terrorist, thug, trash, vermin, worthless
- Novel terms not in baseline (sample): able, actually, af, ago, air, aldrig, alle, alt, altid, altså, amp, anden, andet, andre, argument, asked, away, bad, bare, bedre, began, better, big, bit, black, blev, blevet, blive, bliver, blood, body, brain, bruge, bruger, burde, called, came, car, cause, child, city, come, coming, couldn, course, da, dag, damn, danmark, dansk
