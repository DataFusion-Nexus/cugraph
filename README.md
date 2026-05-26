# DataFusion Nexus cuGraph fork

This repository is forked from NVIDIA RAPIDS cuGraph at `v26.06.00`.

The upstream project is maintained at: https://github.com/rapidsai/cugraph.

This fork keeps the upstream cuGraph codebase as its base and carries local changes needed by
`datafusion-nexus` (check [diff](https://github.com/rapidsai/cugraph/compare/v26.06.00...DataFusion-Nexus:cugraph:datafusion-nexus)).

## Base

- Upstream: NVIDIA RAPIDS cuGraph
- Base tag: `v26.06.00`
- Base commit in this repository: `ec1b6127d4979d9f36e232f6f73537372ebdc7e0`

## Local Changes

This fork currently adds predicate-aware breadth-first search support across the C++, C, and Python
cuGraph layers:

- `cugraph::bfs_with_predicates` for storage-aligned edge predicates, local vertex allow bitmaps,
  target predicates, and level-synchronous target discovery metadata.
- `cugraph_bfs_with_predicates` in the C API, including include/exclude vertex sets, target
  vertices, include edge IDs, optional predecessor output, and explicit target-discovery result
  accessors.
- Python `cugraph.bfs` extensions for `include_vertices`, `exclude_vertices`, `target_vertices`,
  `include_edge_ids`, and `stop_on_first_target`.
- `pylibcugraph.bfs_with_predicates` bindings for the new C API surface and predicate metadata.
- BFS traversal implementation support for predicate filtering, target early-stop behavior, and
  predecessor/distance output compatibility with the existing BFS contract.

The fork includes focused C++, C API, and Python tests for predicate-aware BFS behavior, including
edge filtering, vertex filtering, target discovery, and public Python API coverage.
