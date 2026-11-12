# 016 TODO and FIXME Tracking

## Repository Scan Method
Pattern search executed for `TODO|FIXME|XXX|HACK|DEPRECATED` across source files, excluding [models](../models), notebooks, and `.git` internals.

## Scan Result
No actionable TODO/FIXME markers were found in runtime or script source files.

## Notes on Exclusions
- Model tokenizer and vocabulary artifacts naturally contain lexical tokens such as `todo` or `hack`; these are not developer action markers.
- `.git/hooks` sample files may contain TODO comments unrelated to project code.

## Deprecated Paths
No explicit deprecated code markers were found.

## Operationally Relevant Follow-Up Items (Created from Verified Findings)
1. Introduce authenticated access controls for websocket and status endpoints.
2. Move runtime overrides to connection-local scope.
3. Add payload and frame-size validation in the websocket handler.
4. Reduce `np.concatenate()` allocation churn during audio accumulation.
5. Add tests for classification branch order, fallback behavior, and classifier artifact compatibility.
6. Add explicit cleanup and lifecycle management for helper workers and startup resources.
7. Remove hardcoded service credentials and credential-bearing audio URL defaults from runtime and documentation.

## Risk Summary
- Absence of TODO/FIXME comments does not indicate low debt.
- Debt is present as structural and operational risk documented in [014-known-issues-and-tech-debt.md](014-known-issues-and-tech-debt.md).
