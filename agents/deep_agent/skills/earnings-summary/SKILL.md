---
name: earnings-summary
description: Produce a structured earnings recap for a public company based on research findings. Use when asked to summarize a quarterly result, earnings release, or company report.
---

# Earnings Summary Skill

## Format

A compact table of headline numbers, followed by a short narrative.

| Metric | Result | Consensus | Y/Y |
|---|---|---|---|
| Revenue | … | … | … |
| EPS (GAAP / Adj.) | … | … | … |
| Operating margin | … | … | … |
| Guidance change | … | — | — |

Then 3–5 bullets:
- What drove the print (segments, geographies, one-offs)
- What management called out on the call
- Where the result diverged from expectations

## Rules
- Pull figures only from sourced material; mark unknown values as `n/a`
- Note the reporting period (e.g., "Q3 FY2025")
- Flag any restatements or non-GAAP adjustments explicitly
