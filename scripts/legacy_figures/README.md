# Legacy figure scripts — superseded, do not use for the manuscript

These scripts produced figures for earlier versions of the paper (rounds 1–2).
Several contain synthetic fallbacks (`np.random` placeholder intervals or noise),
hard-coded values, or stale numbers (for example a 47 ms latency). They are kept
only so the history of the earlier figures can be inspected.

Every table, figure and number in the current manuscript is produced by
`paper_build/` from the committed result files in `experiments/`:

    python -m paper_build

`paper_build` refuses to build if it finds random-number use outside its seeded
bootstrap or a typed-in result.
