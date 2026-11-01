# Writer-side sidecar pairing — root cause and fix (LP-0MVVI8TEU006KTOZ)

Work item: LP-0MVVI8TEU006KTOZ (discovered from LP-0MVTJVNSO008IZEB, which
validated fix `5502ee44` and measured **NOT FIXED** over the 2026-10-29/30
post-deploy window).

## Verdict

**Root cause identified and fixed.** The proxy and llama-server disagreed about
*where* a slot snapshot is written:

- `proxy/proxy/session.py:734` sends only the **basename** to llama-server
  (`payload = {"filename": Path(filename).name}`).
- llama-server accepts only a bare filename
  (`fs_validate_filename(..., allow_subdirs=false)`, `common/common.cpp:761`)
  and writes `params.slot_save_path + filename` (the cache **root**,
  `tools/server/server-context.cpp:4053`).
- `bb4c2d44` (LP-0MRPILSMW004T4H8) added a `{host}-{port}/` subdirectory to
  the proxy-side path in `_build_slot_context`, so **only the proxy** believed
  the snapshot was in `slot-cache/localhost-8080/`.

Consequences, all observed: the `.ratio.json` sidecar was written to
`slot-cache/localhost-8080/`, the snapshot to `slot-cache/`; the restore
existence pre-check (`_restore_slot_snapshot`) looked in the subdirectory and
short-circuited **before** any HTTP call (hence 0 `slot_restore failed`); the
seed lookup found the root snapshot and then reported `missing_sidecar`. The
credit could never fire, no matter how good the writer-side code was.

The fix makes the proxy use llama-server's real save directory for save,
restore, sidecar and seed, and keeps the endpoint subdirectory only as a
legacy read path. The eviction path keeps its documented behaviour (a session
with no recorded prefix still saves its snapshot but writes no sidecar — never
an invented count); the extracted `_eviction_sidecar_source` helper makes that
single lookup explicit.

## AC1 — which write path produced each `missing_sidecar` snapshot

Window: `2026-10-29T12:58:32Z` → `2026-10-30T12:58:32Z` (the ≥24 h post-fix
window measured by LP-0MVTJVNSO008IZEB).

Exact command (JSON artifact committed next to this file):

```bash
python3 scripts/attribute_missing_sidecar.py \
  --log-dir /var/log/llama-proxy \
  --window "2026-10-29T12:58:32,2026-10-30T12:58:32" \
  --json > docs/dev/slot-sidecar-attribution-LP-0MVVI8TEU006KTOZ.json
```

Seed-outcome vocabulary for the same window (existing tool, unchanged):

```bash
python3 scripts/analyse_slot_ratio_restore.py \
  --log-dir /var/log/llama-proxy \
  --window "post-fix=2026-10-29T12:58:32,2026-10-30T12:58:32" \
  --json
```

Result (`slot-sidecar-attribution-LP-0MVVI8TEU006KTOZ.json`):

| Metric | Count |
|---|---|
| `slot_ratio_seed outcome=missing_sidecar` | **605** |
| `slot_ratio_seed outcome=missing_snapshot` | 12 |
| `outcome=found` | **0** |
| `cache_state=restored` dispatches | **0** |
| Router-path saves succeeded | 34 |
| Eviction-path saves succeeded | 42 |
| Eviction saves failed (`saved=False`, no pairing marker) | 64 |
| `slot persistence skipped reason=slot_busy` | 260 |
| Distinct sessions reporting `missing_sidecar` | 9 |

Attribution of every one of the 605 `missing_sidecar` seeds to the most recent
successful save for that session:

| Write path | Seeds |
|---|---|
| Eviction (`slot_save … saved=True source=eviction`) | **414** |
| Router (`slot_save success …`) | **181** |
| No successful save in the window | 10 |

Per session (top 5): `audit-AH` 204 (router 118 / eviction 86), `audit-LP` 153
(eviction 153), `herdr-17` 79 (eviction 63 / router 15 / none 1), `audit-SA` 74
(eviction 54 / router 20), `audit-RI` 54 (eviction 33 / router 21). Both writer
paths were therefore affected, which rules out a single-path bug and matches a
path-resolution defect shared by both.

**On-disk proof** (taken while the pre-fix proxy was still running):

```bash
cd /home/rgardler/projects/llm/slot-cache
echo "snapshots=$(find . -name '*.bin' | wc -l) sidecars=$(find . -name '*.bin.ratio.json' | wc -l)"
for f in *.bin; do
  echo "$f root-sidecar=$([ -f "$f.ratio.json" ] && echo yes || echo NO) \
endpoint-sidecar=$([ -f "localhost-8080/$f.ratio.json" ] && echo yes || echo no)"
done
```

Every root `slot_*.bin` reported `root-sidecar=NO endpoint-sidecar=yes`
(e.g. `slot_audit-RISE-0MVPY83CA00335W3-phase2_deep-e4bd77bb.bin` written at
22:52Z with its `.ratio.json` under `localhost-8080/`). The live proxy
reproduced the split on every save: `.bin` at the root, `.ratio.json` one
directory down, same basename, same second.

## Root cause

`session._build_slot_context` returned

```text
<session_slot_save_path>/<host>-<port>/slot_<session>.bin
```

for a request carrying an `endpoint`, but `session._call_slot_endpoint` sent
llama-server only `slot_<session>.bin`; the server resolved it against its own
`--slot-save-path` (`/home/rgardler/projects/llm/slot-cache`, `models.ini:17`;
also exported as `LLAMA_SLOT_SAVE_PATH`, `proxy/proxy/lifecycle.py:1636`). The
snapshot was therefore at `<root>/slot_<session>.bin` while the sidecar,
`sidecar_path(slot_filename)`, and the restore pre-check were at
`<root>/localhost-8080/…`. `_candidate_slot_filenames` tried the subdirectory
first, found no snapshot there, fell back to the root — and then required the
sidecar next to the root snapshot, which never existed. `missing_sidecar` on
every request; `_restore_slot_snapshot`'s `Path(filename).exists()` returned
False, so restore never ran.

The earlier tests masked it: their `_save_slot_snapshot` stubs wrote the
snapshot at the proxy-side path instead of reproducing llama-server's
basename-under-`--slot-save-path` contract.

## Fix

1. `session._build_slot_context` (`proxy/proxy/session.py`): the snapshot path
   is the shared `session_slot_save_path` root; `endpoint` still selects which
   llama-server receives the save/restore, but no longer changes the directory.
2. `slot_cache_ratio._candidate_slot_filenames`: root first, then a legacy
   `{host}-{port}/` candidate, so existing snapshots/sidecars stay discoverable
   and retire normally.
3. `session._eviction_sidecar_source`: the eviction path's recorded-prefix
   lookup is extracted into one helper used by `_write_eviction_sidecar`, so
   the pairing source is resolved in one place. Behaviour is unchanged: an
   eviction save with no recorded prefix still writes its snapshot but no
   sidecar (documented in LP-0MVK0E80C009HUQC — never invent a token count),
   and the session's next measured turn writes a paired snapshot on the router
   path.
4. Config/docs (`proxy/config.yaml`, `proxy/README.md`, `proxy/MODEL_ADD.md`)
   now state that every local llama-server's `--slot-save-path` must point at
   `session_slot_save_path`, and that the `{host}-{port}/` layout is legacy.

No credit-policy change: `prefix_matches`, the TTL, the safety margin, the
sidecar schema and the `slot_ratio_seed`/`slot_ratio_restored` log formats are
untouched (the existing guard tests stay green).

**Migration note.** Root snapshots saved before the fix have no root sidecar;
they stay `missing_sidecar` until the session saves once through the fixed
code (which writes the sidecar next to the snapshot). The legacy
`{host}-{port}/` sidecars are still retired with their session and pruned by
`scripts/cleanup-slot-cache.sh`.

**Known limitation (unchanged on disk).** The on-disk filename is
session-scoped, so the same client-supplied session id used against two local
backends shares one snapshot file. That was already true before the fix (every
snapshot landed in the root); restore still targets the endpoint that served
the request. True per-endpoint isolation would need an
`--slot-save-path` per llama-server (deployment-level) or an endpoint
 discriminator in the basename.

## Tests

- `proxy/tests/test_slot_cache_ratio_restore.py`
  - `TestSnapshotDirectoryAlignment::test_slot_context_uses_the_llama_server_save_directory`
    — an endpoint no longer moves the snapshot out of the root.
  - `TestSnapshotDirectoryAlignment::test_router_save_is_seedable_where_llama_server_wrote_it`
    — end-to-end regression: the stub reproduces llama-server's
    basename-under-`--slot-save-path` write, and the save then seeds
    (`outcome=found`, `cached_ratio = 0.9 × margin`). This test fails on the
    pre-fix code (captured red run: 3 failures) and passes after the fix.
  - `TestEvictionSnapshotPairing::test_no_recorded_prefix_writes_no_sidecar`
    — the eviction path keeps its documented no-invented-count behaviour.
  - `TestSeedPath::test_legacy_endpoint_directory_snapshot_still_seeds` —
    backward compatibility for the legacy layout.
- `proxy/tests/test_multi_backend.py` — AC3 assertion updated from
  "per-endpoint subdirectory" to the shared root (the old assertion encoded
  the defect).
- `tests/test_attribute_missing_sidecar.py` — the attribution tool: per-path
  counts, most-recent-save wins, window filtering, `.gz` logs, failed-save vs
  pairing-skip, empty window.

## AC3 — post-deploy ≥24 h window (deferred, not claimed)

AC3 requires a ≥24 h window **after the new deploy** with `outcome=found` > 0
and `cache_state=restored` > 0, or a documented proof that the credit cannot
fire. The credit is expected to fire once this fix is deployed, so the
impossibility branch does not apply; and the ≥24 h window cannot exist before
the fix is deployed. Following the LP-0MVK0E80C009HUQC → LP-0MVTJVNSO008IZEB
pattern, the time-gated measurement is tracked as a separate follow-up item:
**LP-0MVVLQMG1000843Q** ("Validate writer-side sidecar pairing fix — >=24 h
post-deploy window", scheduling hold until deploy + 24 h).

Expected post-deploy signature (for that follow-up):

- `slot_ratio_seed outcome=found` > 0 and `slot_ratio_restored` > 0;
- `cache_state=restored` dispatches > 0;
- `missing_sidecar` near 0 for sessions saved after the deploy;
- `missing_snapshot` may appear for legacy orphan sidecars (cleared by
  retirement/`scripts/cleanup-slot-cache.sh`), and `prefix_mismatch` /
  `expired` become possible — those are the **success** signal that the guard
  now runs, not a regression.

## Reproduce

```bash
# AC1 attribution (JSON artifact)
python3 scripts/attribute_missing_sidecar.py \
  --log-dir /var/log/llama-proxy \
  --window "2026-10-29T12:58:32,2026-10-30T12:58:32" --json

# Seed-outcome tally (existing tool)
python3 scripts/analyse_slot_ratio_restore.py \
  --log-dir /var/log/llama-proxy \
  --window "post-fix=2026-10-29T12:58:32,2026-10-30T12:58:32" --json

# On-disk pairing (after the fix, every pair must share a directory)
cd /home/rgardler/projects/llm/slot-cache
find . -name '*.bin' -o -name '*.bin.ratio.json' | sort
```

Artifacts:

- `docs/dev/slot-sidecar-attribution-LP-0MVVI8TEU006KTOZ.json`
- `scripts/attribute_missing_sidecar.py` + `tests/test_attribute_missing_sidecar.py`
