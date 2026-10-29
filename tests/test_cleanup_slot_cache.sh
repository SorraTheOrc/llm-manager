#!/usr/bin/env bash
# Tests for scripts/cleanup-slot-cache.sh
# Uses temporary directories to avoid touching the real slot-cache.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
SCRIPT="$SCRIPT_DIR/cleanup-slot-cache.sh"
TESTS_TMPDIR=""
PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

cleanup_tmp() {
    if [ -n "$TESTS_TMPDIR" ] && [ -d "$TESTS_TMPDIR" ]; then
        rm -rf "$TESTS_TMPDIR"
    fi
    TESTS_TMPDIR=""
}

# Create a slot file with a given age (in days) and size (in MB)
make_slot() {
    local dir="$1"
    local age_days="$2"
    local size_mb="${3:-1}"
    local uuid="${4:-$(uuidgen 2>/dev/null || echo "00000000-0000-0000-0000-000000000000")}"
    mkdir -p "$dir"
    local file="$dir/slot_${uuid}.bin"
    truncate -s "${size_mb}M" "$file"
    # Set mtime to `age_days` days ago
    touch -d "$(date -d "$age_days days ago" +%Y-%m-%dT%H:%M:%S)" "$file"
}

# Create a sidecar next to a slot file (matching age)
make_sidecar() {
    local file="$1"
    echo '{}' > "${file}.ratio.json"
    if [ -f "$file" ]; then
        touch -r "$file" "${file}.ratio.json"
    fi
}

# ---------------------------------------------------------------
# Test: --dry-run lists candidates without deleting anything
# ---------------------------------------------------------------
test_dry_run() {
    echo "Test: --dry-run previews deletions and deletes nothing"
    TESTS_TMPDIR="$(mktemp -d)"
    local cache="$TESTS_TMPDIR/cache"
    make_slot "$cache" 10 1 "aaaaaaaa-0000-0000-0000-000000000000"   # old
    make_slot "$cache" 1 1  "bbbbbbbb-0000-0000-0000-000000000000"   # recent
    local before
    before=$(find "$cache" -name '*.bin' | wc -l)

    local out
    out=$("$SCRIPT" --path "$cache" --max-age-days 7 --max-size-gb 100 --dry-run 2>&1)
    local after
    after=$(find "$cache" -name '*.bin' | wc -l)

    if echo "$out" | grep -q "Would delete"; then pass "dry-run lists candidates"; else fail "dry-run listed no candidates"; fi
    if [ "$before" = "$after" ]; then pass "dry-run deleted nothing"; else fail "dry-run deleted files (before=$before after=$after)"; fi
    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: age rule deletes old files and keeps recent ones
# ---------------------------------------------------------------
test_age_rule() {
    echo "Test: files older than --max-age-days are deleted, recent kept"
    TESTS_TMPDIR="$(mktemp -d)"
    local cache="$TESTS_TMPDIR/cache"
    make_slot "$cache" 10 1 "aaaaaaaa-0000-0000-0000-000000000000"   # 10 days old
    make_slot "$cache" 1 1  "bbbbbbbb-0000-0000-0000-000000000000"   # 1 day old

    "$SCRIPT" --path "$cache" --max-age-days 7 --max-size-gb 100 >/dev/null 2>&1

    if [ ! -f "$cache/slot_aaaaaaaa-0000-0000-0000-000000000000.bin" ]; then
        pass "old file deleted"
    else
        fail "old file still present"
    fi
    if [ -f "$cache/slot_bbbbbbbb-0000-0000-0000-000000000000.bin" ]; then
        pass "recent file retained"
    else
        fail "recent file was deleted"
    fi
    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: size cap prunes oldest beyond cap but protects active sessions
# ---------------------------------------------------------------
test_size_cap() {
    echo "Test: --max-size-gb prunes oldest files, protecting files newer than --min-age-days"
    TESTS_TMPDIR="$(mktemp -d)"
    local cache="$TESTS_TMPDIR/cache"
    # 15 files x 100MB = 1.5GB; cap 1GB => prune 5 oldest
    for i in $(seq 1 15); do
        make_slot "$cache" $((20 - i)) 100 "00000000-0000-0000-0000-$(printf '%012d' $i)"
    done

    "$SCRIPT" --path "$cache" --max-age-days 30 --max-size-gb 1 --min-age-days 1 >/dev/null 2>&1

    local remaining
    remaining=$(find "$cache" -name '*.bin' | wc -l)
    if [ "$remaining" = "10" ]; then pass "pruned to 10 files (oldest removed)"; else fail "expected 10 files, found $remaining"; fi
    # The 5 most recent files (ages 5..1 days) must survive; oldest (ages 19..15) must be gone
    if [ -f "$cache/slot_00000000-0000-0000-0000-000000000015.bin" ]; then pass "newest retained"; else fail "newest pruned"; fi
    if [ ! -f "$cache/slot_00000000-0000-0000-0000-000000000001.bin" ]; then pass "oldest pruned"; else fail "oldest retained"; fi
    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: active-session protection - files newer than --min-age-days survive
# ---------------------------------------------------------------
test_active_protection() {
    echo "Test: files newer than --min-age-days are never deleted even over cap"
    TESTS_TMPDIR="$(mktemp -d)"
    local cache="$TESTS_TMPDIR/cache"
    # 2 files at 0 days (protected) + 11 files at 5 days = 1.3GB; cap 1GB
    make_slot "$cache" 0 100 "aaaaaaaa-0000-0000-0000-000000000000"
    make_slot "$cache" 0 100 "bbbbbbbb-0000-0000-0000-000000000000"
    for i in $(seq 1 11); do
        make_slot "$cache" 5 100 "cccccccc-0000-0000-0000-$(printf '%012d' $i)"
    done

    "$SCRIPT" --path "$cache" --max-age-days 30 --max-size-gb 1 --min-age-days 1 >/dev/null 2>&1

    local remaining
    remaining=$(find "$cache" -name '*.bin' | wc -l)
    if [ "$remaining" = "10" ]; then pass "pruned to cap (10 files)"; else fail "expected 10 files, found $remaining"; fi
    if [ -f "$cache/slot_aaaaaaaa-0000-0000-0000-000000000000.bin" ] && [ -f "$cache/slot_bbbbbbbb-0000-0000-0000-000000000000.bin" ]; then
        pass "active (recent) files retained"
    else
        fail "active files were deleted"
    fi
    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: no-op when cache is under cap and all files recent
# ---------------------------------------------------------------
test_noop_when_healthy() {
    echo "Test: nothing deleted when cache is under cap and all files recent"
    TESTS_TMPDIR="$(mktemp -d)"
    local cache="$TESTS_TMPDIR/cache"
    make_slot "$cache" 1 1 "aaaaaaaa-0000-0000-0000-000000000000"
    make_slot "$cache" 2 1 "bbbbbbbb-0000-0000-0000-000000000000"

    local out
    out=$("$SCRIPT" --path "$cache" --max-age-days 7 --max-size-gb 100 2>&1)
    local remaining
    remaining=$(find "$cache" -name '*.bin' | wc -l)

    if [ "$remaining" = "2" ]; then pass "all files retained"; else fail "files deleted in healthy cache"; fi
    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: a deleted snapshot takes its sidecar with it (no orphan)
# ---------------------------------------------------------------
test_pair_deleted_together() {
    echo "Test: deleting a snapshot also deletes its .ratio.json sidecar"
    TESTS_TMPDIR="$(mktemp -d)"
    local cache="$TESTS_TMPDIR/cache"
    make_slot "$cache" 10 1 "aaaaaaaa-0000-0000-0000-000000000000"   # old
    make_sidecar "$cache/slot_aaaaaaaa-0000-0000-0000-000000000000.bin"
    make_slot "$cache" 1 1 "bbbbbbbb-0000-0000-0000-000000000000"    # recent
    make_sidecar "$cache/slot_bbbbbbbb-0000-0000-0000-000000000000.bin"

    "$SCRIPT" --path "$cache" --max-age-days 7 --max-size-gb 100 >/dev/null 2>&1

    if [ ! -f "$cache/slot_aaaaaaaa-0000-0000-0000-000000000000.bin.ratio.json" ]; then
        pass "sidecar deleted with its snapshot"
    else
        fail "sidecar outlived its snapshot"
    fi
    if [ -f "$cache/slot_bbbbbbbb-0000-0000-0000-000000000000.bin.ratio.json" ]; then
        pass "sidecar of the retained snapshot survived"
    else
        fail "sidecar of a retained snapshot was deleted"
    fi
    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: endpoint subdirectories are pruned, not just the root
# ---------------------------------------------------------------
test_endpoint_subdirectories() {
    echo "Test: per-endpoint subdirectories are included in the scan"
    TESTS_TMPDIR="$(mktemp -d)"
    local cache="$TESTS_TMPDIR/cache"
    mkdir -p "$cache/localhost-8080"
    make_slot "$cache/localhost-8080" 10 1 "aaaaaaaa-0000-0000-0000-000000000000"
    make_sidecar "$cache/localhost-8080/slot_aaaaaaaa-0000-0000-0000-000000000000.bin"
    make_slot "$cache/localhost-8080" 1 1 "bbbbbbbb-0000-0000-0000-000000000000"
    make_sidecar "$cache/localhost-8080/slot_bbbbbbbb-0000-0000-0000-000000000000.bin"

    "$SCRIPT" --path "$cache" --max-age-days 7 --max-size-gb 100 >/dev/null 2>&1

    if [ ! -f "$cache/localhost-8080/slot_aaaaaaaa-0000-0000-0000-000000000000.bin" ]; then
        pass "old snapshot in a subdirectory deleted"
    else
        fail "old snapshot in a subdirectory retained (root-only glob?)"
    fi
    if [ -f "$cache/localhost-8080/slot_bbbbbbbb-0000-0000-0000-000000000000.bin" ]; then
        pass "recent snapshot in a subdirectory retained"
    else
        fail "recent snapshot in a subdirectory was deleted"
    fi
    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: orphan sidecars are pruned even with no snapshots at all
# ---------------------------------------------------------------
test_orphan_sidecars_pruned() {
    echo "Test: orphan .ratio.json files are pruned (root and subdirectories)"
    TESTS_TMPDIR="$(mktemp -d)"
    local cache="$TESTS_TMPDIR/cache"
    mkdir -p "$cache/localhost-8080"
    echo '{}' > "$cache/slot_root-orphan.bin.ratio.json"
    echo '{}' > "$cache/localhost-8080/slot_nested-orphan.bin.ratio.json"

    local rc
    "$SCRIPT" --path "$cache" --max-age-days 7 --max-size-gb 100 >/dev/null 2>&1
    rc=$?

    if [ "$rc" = "0" ]; then pass "script exits 0 with only orphan sidecars"; else fail "script exited $rc"; fi
    if [ ! -f "$cache/slot_root-orphan.bin.ratio.json" ]; then
        pass "root orphan sidecar pruned"
    else
        fail "root orphan sidecar survived"
    fi
    if [ ! -f "$cache/localhost-8080/slot_nested-orphan.bin.ratio.json" ]; then
        pass "nested orphan sidecar pruned"
    else
        fail "nested orphan sidecar survived"
    fi
    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: missing directory exits non-zero
# ---------------------------------------------------------------
test_missing_dir() {
    echo "Test: missing directory exits with error"
    local out rc
    out=$("$SCRIPT" --path /nonexistent/slot-cache 2>&1)
    rc=$?
    if [ "$rc" != "0" ]; then pass "missing dir exits $rc"; else fail "missing dir exited 0"; fi
    cleanup_tmp
}

# Run all tests
test_dry_run
test_age_rule
test_size_cap
test_active_protection
test_noop_when_healthy
test_pair_deleted_together
test_endpoint_subdirectories
test_orphan_sidecars_pruned
test_missing_dir

echo ""
echo "RESULTS: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
