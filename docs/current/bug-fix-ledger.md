# Bug-fix ledger

Live status of the defects found in the old code. The full descriptions and fixes are in
[../rewrite-plan.md](../rewrite-plan.md) §8.

A bug is **closed** only when its fix is merged and its regression test passes.

| ID | Sev | Phase | Status | Regression test |
|---|---|---|---|---|
| B1 | High | M1/M3/M4/M6 | Open | `test_cache_key_invalidation_matrix` |
| B2 | High | M2 | Open | `test_integrity_checks_measured_files` |
| B3 | High | M1 | Open | `test_offline_error_matches_formula` |
| B4 | High | M1/M3 | Open | `test_score_interval_under_dependence` |
| B5 | Med | M4 | Open | `test_first_look_cut_from_se` |
| B6 | Med | M4 | Open | `test_patience_does_not_ratchet` |
| B7 | Med | M4 | Open | `test_solver_stability` |
| B8 | Med | M4 | Open | `test_ga_is_deterministic` |
| B9 | High | M6 | Open | `test_claim_caps` |
| B10 | Med | M5 | Open | `test_disk_preflight` |
| B11 | Med | M5/M10 | Open | `test_24b_load` (gpu) |
| B12 | Med | M3 | Open | `test_server_port_race` |
| B13 | Low | M3/M6 | Open | `test_thread_policy` |
| B14 | Low | M6/M7 | Open | `test_caps_have_one_source` |
| B15 | Low | M1/M4 | Open | `test_preset_bpw_table` |
| B16 | Low | M6 | Open | `test_inflight_planner` |
| B17 | Low | M3 | Open | `test_gsm8k_cache_key` |
| B18 | Low | M5 | Open | `test_rounding_plan` |
| B19 | Low | M2 | Open | `test_manifest_completeness` |
| B20 | Low | M0 | **Closed** | fixed by the root `.gitignore`/`.containerignore` rewrite |
| B21 | Low | M0 | **Closed** | old code not carried over; lint prevents reintroduction |
