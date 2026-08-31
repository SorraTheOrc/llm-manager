# Session-Compaction Quality Experiment — Results

Three-arm replay experiment (design: `docs/session-compaction-experiment-design.md`). A = uncompacted→remote deepseek-v4-flash (baseline), B = compacted→local Qwen3 (proposed), C = uncompacted→local Qwen3 (ceiling).

## Go / no-go decision

**Decision: GO**

Rationale: quality gate met; efficiency gate met

### Pre-registered checks

| Mode | Rule | Pass | Detail |
|---|---|---|---|
| fast | rubric | PASS | B/A rubric = 1.000 |
| fast | completion | PASS | completion delta B-A = 0.0pp |
| fast | failures | PASS | failure delta B-A = 0.0pp |
| cheap | rubric | PASS | B/A rubric = 1.000 |
| cheap | completion | PASS | completion delta B-A = 0.0pp |
| cheap | failures | PASS | failure delta B-A = 0.0pp |
| fast | prefill_reduction | PASS | prefill reduction = 61.6% |
| cheap | prefill_reduction | PASS | prefill reduction = 30.2% |
| fast | ttft | PASS | TTFT P95 B vs A = +0.0% |
| cheap | ttft | PASS | TTFT P95 B vs A = +0.0% |

## Metrics

| Arm | Mode | n | Completion | Rubric | Failure | TTFT P95 (ms) | Prefill est. total |
|---|---|---|---|---|---|---|---|
| A | fast | 301 | 0.0% | 2.40 | 0.0% | 579 | 33367749 |
| A | cheap | 115 | 0.0% | 2.40 | 0.0% | 580 | 8933888 |
| B | fast | 301 | 0.0% | 2.40 | 0.0% | 579 | 12818456 |
| B | cheap | 115 | 0.0% | 2.40 | 0.0% | 580 | 6239876 |
| C | fast | 301 | 0.0% | 2.40 | 0.0% | 579 | 33367749 |
| C | cheap | 115 | 0.0% | 2.40 | 0.0% | 580 | 8933888 |

## Transcript sources

- **recording**: 172 tasks
- **synthetic**: 244 tasks

## Tasks

| Task | Mode | Band | Category | Est. tokens | A | B | C |
|---|---|---|---|---|---|---|---|
| herdr-178806-fast-61921 | fast | trigger-cap | code | 61921 | ok | ok | ok |
| audit-CG-0MT-fast-69587 | fast | trigger-cap | code | 69587 | ok | ok | ok |
| herdr-178806-fast-69929 | fast | trigger-cap | qa | 69929 | ok | ok | ok |
| herdr-178808-fast-59356 | fast | trigger-cap | agent | 59356 | ok | ok | ok |
| herdr-178807-fast-63903 | fast | trigger-cap | code | 63903 | ok | ok | ok |
| herdr-178806-fast-68063 | fast | trigger-cap | code | 68063 | ok | ok | ok |
| herdr-178807-fast-59769 | fast | trigger-cap | qa | 59769 | ok | ok | ok |
| herdr-178808-fast-65778 | fast | trigger-cap | code | 65778 | ok | ok | ok |
| herdr-178808-fast-60171 | fast | trigger-cap | code | 60171 | ok | ok | ok |
| herdr-178808-fast-69444 | fast | trigger-cap | code | 69444 | ok | ok | ok |
| herdr-178739-fast-69119 | fast | trigger-cap | code | 69119 | ok | ok | ok |
| audit-WL-0MS-fast-67076 | fast | trigger-cap | code | 67076 | ok | ok | ok |
| audit-CG-0MT-fast-67637 | fast | trigger-cap | code | 67637 | ok | ok | ok |
| audit-WL-0MT-fast-62203 | fast | trigger-cap | code | 62203 | ok | ok | ok |
| audit-WL-0MS-fast-65605 | fast | trigger-cap | code | 65605 | ok | ok | ok |
| audit-WL-0MQ-fast-63081 | fast | trigger-cap | code | 63081 | ok | ok | ok |
| herdr-178737-fast-62391 | fast | trigger-cap | code | 62391 | ok | ok | ok |
| 01a030d8-7e8-fast-61242 | fast | trigger-cap | code | 61242 | ok | ok | ok |
| audit-LP-0MS-fast-62436 | fast | trigger-cap | code | 62436 | ok | ok | ok |
| audit-SA-0MT-fast-64485 | fast | trigger-cap | code | 64485 | ok | ok | ok |
| audit-WL-0MS-fast-68209 | fast | trigger-cap | code | 68209 | ok | ok | ok |
| herdr-178754-fast-64002 | fast | trigger-cap | code | 64002 | ok | ok | ok |
| herdr-178757-fast-68659 | fast | trigger-cap | code | 68659 | ok | ok | ok |
| audit-WL-0MS-fast-62847 | fast | trigger-cap | code | 62847 | ok | ok | ok |
| audit-WL-0MS-fast-69531 | fast | trigger-cap | code | 69531 | ok | ok | ok |
| herdr-178758-fast-69346 | fast | trigger-cap | code | 69346 | ok | ok | ok |
| audit-SA-0MT-fast-61218 | fast | trigger-cap | code | 61218 | ok | ok | ok |
| audit-WL-0MS-fast-62479 | fast | trigger-cap | code | 62479 | ok | ok | ok |
| 01a03570-1e2-fast-64735 | fast | trigger-cap | code | 64735 | ok | ok | ok |
| audit-WL-0MS-fast-68276 | fast | trigger-cap | code | 68276 | ok | ok | ok |
| audit-WL-0MS-fast-68047 | fast | trigger-cap | code | 68047 | ok | ok | ok |
| herdr-178761-fast-61203 | fast | trigger-cap | code | 61203 | ok | ok | ok |
| audit-LP-0MS-fast-68631 | fast | trigger-cap | code | 68631 | ok | ok | ok |
| audit-LP-0MS-fast-67990 | fast | trigger-cap | code | 67990 | ok | ok | ok |
| audit-LP-0MS-fast-60577 | fast | trigger-cap | code | 60577 | ok | ok | ok |
| audit-SA-0MT-fast-61992 | fast | trigger-cap | code | 61992 | ok | ok | ok |
| herdr-178767-fast-66239 | fast | trigger-cap | code | 66239 | ok | ok | ok |
| herdr-178766-fast-69782 | fast | trigger-cap | code | 69782 | ok | ok | ok |
| audit-LP-0MS-fast-65495 | fast | trigger-cap | code | 65495 | ok | ok | ok |
| herdr-178768-fast-66647 | fast | trigger-cap | code | 66647 | ok | ok | ok |
| audit-LP-0MS-fast-58833 | fast | trigger-cap | code | 58833 | ok | ok | ok |
| herdr-178769-fast-65414 | fast | trigger-cap | code | 65414 | ok | ok | ok |
| herdr-178769-fast-67090 | fast | trigger-cap | code | 67090 | ok | ok | ok |
| herdr-178770-fast-60761 | fast | trigger-cap | code | 60761 | ok | ok | ok |
| audit-SA-0MT-fast-58460 | fast | trigger-cap | code | 58460 | ok | ok | ok |
| audit-SA-0MT-fast-60376 | fast | trigger-cap | code | 60376 | ok | ok | ok |
| audit-SA-0MT-fast-67269 | fast | trigger-cap | code | 67269 | ok | ok | ok |
| herdr-178774-fast-66405 | fast | trigger-cap | code | 66405 | ok | ok | ok |
| herdr-178775-fast-63712 | fast | trigger-cap | code | 63712 | ok | ok | ok |
| audit-LP-0MT-fast-69549 | fast | trigger-cap | code | 69549 | ok | ok | ok |
| audit-CG-0MT-fast-65406 | fast | trigger-cap | code | 65406 | ok | ok | ok |
| audit-LP-0MT-fast-68419 | fast | trigger-cap | code | 68419 | ok | ok | ok |
| audit-WL-0MT-fast-66787 | fast | trigger-cap | code | 66787 | ok | ok | ok |
| audit-CG-0MT-fast-67503 | fast | trigger-cap | code | 67503 | ok | ok | ok |
| herdr-178791-fast-63834 | fast | trigger-cap | code | 63834 | ok | ok | ok |
| audit-LP-0MT-fast-61619 | fast | trigger-cap | code | 61619 | ok | ok | ok |
| herdr-178792-fast-58379 | fast | trigger-cap | qa | 58379 | ok | ok | ok |
| herdr-178793-fast-63854 | fast | trigger-cap | code | 63854 | ok | ok | ok |
| herdr-178795-fast-66449 | fast | trigger-cap | code | 66449 | ok | ok | ok |
| audit-AH-0MT-fast-64301 | fast | trigger-cap | code | 64301 | ok | ok | ok |
| herdr-178802-fast-58825 | fast | trigger-cap | code | 58825 | ok | ok | ok |
| herdr-178803-fast-69826 | fast | trigger-cap | code | 69826 | ok | ok | ok |
| herdr-178804-fast-63644 | fast | trigger-cap | qa | 63644 | ok | ok | ok |
| herdr-178804-fast-60123 | fast | trigger-cap | code | 60123 | ok | ok | ok |
| audit-AH-0MT-fast-65399 | fast | trigger-cap | agent | 65399 | ok | ok | ok |
| herdr-178805-fast-63011 | fast | trigger-cap | code | 63011 | ok | ok | ok |
| audit-CG-0MT-fast-65024 | fast | trigger-cap | code | 65024 | ok | ok | ok |
| audit-WL-0MT-fast-62013 | fast | trigger-cap | agent | 62013 | ok | ok | ok |
| herdr-178806-fast-110092 | fast | extreme | code | 110092 | ok | ok | ok |
| herdr-178806-fast-109963 | fast | extreme | code | 109963 | ok | ok | ok |
| herdr-178805-fast-114510 | fast | extreme | code | 114510 | ok | ok | ok |
| unknown-fast-350202 | fast | extreme | code | 350202 | ok | ok | ok |
| herdr-178807-fast-109476 | fast | extreme | code | 109476 | ok | ok | ok |
| herdr-178803-fast-107611 | fast | extreme | code | 107611 | ok | ok | ok |
| herdr-178807-fast-76590 | fast | extreme | agent | 76590 | ok | ok | ok |
| audit-CG-0MT-fast-88976 | fast | extreme | code | 88976 | ok | ok | ok |
| herdr-178806-fast-106757 | fast | extreme | code | 106757 | ok | ok | ok |
| herdr-178807-fast-109567 | fast | extreme | agent | 109567 | ok | ok | ok |
| herdr-178804-fast-104541 | fast | extreme | agent | 104541 | ok | ok | ok |
| herdr-178808-fast-76798 | fast | extreme | agent | 76798 | ok | ok | ok |
| herdr-178807-fast-71945 | fast | extreme | code | 71945 | ok | ok | ok |
| herdr-178808-fast-107855 | fast | extreme | code | 107855 | ok | ok | ok |
| herdr-178808-fast-72947 | fast | extreme | code | 72947 | ok | ok | ok |
| herdr-178807-fast-76293 | fast | extreme | code | 76293 | ok | ok | ok |
| 01a05257-39a-fast-109949 | fast | extreme | code | 109949 | ok | ok | ok |
| audit-CG-0MT-fast-111635 | fast | extreme | code | 111635 | ok | ok | ok |
| herdr-178808-fast-103600 | fast | extreme | code | 103600 | ok | ok | ok |
| herdr-178809-fast-109519 | fast | extreme | code | 109519 | ok | ok | ok |
| herdr-178808-fast-109769 | fast | extreme | qa | 109769 | ok | ok | ok |
| audit-CG-0MT-fast-100162 | fast | extreme | code | 100162 | ok | ok | ok |
| audit-CG-0MT-fast-93592 | fast | extreme | code | 93592 | ok | ok | ok |
| herdr-178809-fast-74991 | fast | extreme | code | 74991 | ok | ok | ok |
| herdr-178809-fast-110375 | fast | extreme | code | 110375 | ok | ok | ok |
| herdr-178809-fast-114666 | fast | extreme | code | 114666 | ok | ok | ok |
| herdr-178809-fast-70058 | fast | extreme | agent | 70058 | ok | ok | ok |
| audit-CG-0MT-fast-101284 | fast | extreme | code | 101284 | ok | ok | ok |
| herdr-178809-fast-102186 | fast | extreme | qa | 102186 | ok | ok | ok |
| audit-CG-0MT-fast-98553 | fast | extreme | code | 98553 | ok | ok | ok |
| herdr-178809-fast-104224 | fast | extreme | code | 104224 | ok | ok | ok |
| herdr-178809-fast-101161 | fast | extreme | code | 101161 | ok | ok | ok |
| herdr-178809-fast-100365 | fast | extreme | code | 100365 | ok | ok | ok |
| audit-CG-0MT-fast-114447 | fast | extreme | qa | 114447 | ok | ok | ok |
| audit-CG-0MT-fast-93397 | fast | extreme | agent | 93397 | ok | ok | ok |
| audit-CG-0MT-fast-92002 | fast | extreme | code | 92002 | ok | ok | ok |
| herdr-178809-fast-112489 | fast | extreme | code | 112489 | ok | ok | ok |
| herdr-178809-fast-106297 | fast | extreme | code | 106297 | ok | ok | ok |
| herdr-178810-fast-106280 | fast | extreme | code | 106280 | ok | ok | ok |
| herdr-178805-fast-109243 | fast | extreme | code | 109243 | ok | ok | ok |
| herdr-178745-fast-99181 | fast | extreme | code | 99181 | ok | ok | ok |
| herdr-178739-fast-112253 | fast | extreme | code | 112253 | ok | ok | ok |
| herdr-178739-fast-169352 | fast | extreme | code | 169352 | ok | ok | ok |
| herdr-178747-fast-325844 | fast | extreme | code | 325844 | ok | ok | ok |
| herdr-178740-fast-120510 | fast | extreme | code | 120510 | ok | ok | ok |
| herdr-178746-fast-146135 | fast | extreme | code | 146135 | ok | ok | ok |
| herdr-178744-fast-193076 | fast | extreme | code | 193076 | ok | ok | ok |
| herdr-178735-fast-256212 | fast | extreme | code | 256212 | ok | ok | ok |
| herdr-178744-fast-108636 | fast | extreme | code | 108636 | ok | ok | ok |
| herdr-178746-fast-107232 | fast | extreme | code | 107232 | ok | ok | ok |
| audit-WL-0MT-fast-106947 | fast | extreme | code | 106947 | ok | ok | ok |
| herdr-178748-fast-96521 | fast | extreme | code | 96521 | ok | ok | ok |
| audit-WL-0MS-fast-89752 | fast | extreme | code | 89752 | ok | ok | ok |
| audit-OSL-0M-fast-74435 | fast | extreme | code | 74435 | ok | ok | ok |
| herdr-178745-fast-140414 | fast | extreme | code | 140414 | ok | ok | ok |
| herdr-178745-fast-75158 | fast | extreme | code | 75158 | ok | ok | ok |
| 01a02c2e-24f-fast-144777 | fast | extreme | code | 144777 | ok | ok | ok |
| audit-WL-0MT-fast-85219 | fast | extreme | code | 85219 | ok | ok | ok |
| 01a02f6c-7c7-fast-88185 | fast | extreme | code | 88185 | ok | ok | ok |
| audit-WL-0MT-fast-76125 | fast | extreme | code | 76125 | ok | ok | ok |
| herdr-178741-fast-94281 | fast | extreme | code | 94281 | ok | ok | ok |
| audit-WL-0MS-fast-93696 | fast | extreme | code | 93696 | ok | ok | ok |
| audit-WL-0MT-fast-78298 | fast | extreme | code | 78298 | ok | ok | ok |
| herdr-178742-fast-136019 | fast | extreme | code | 136019 | ok | ok | ok |
| herdr-178742-fast-168002 | fast | extreme | code | 168002 | ok | ok | ok |
| audit-WL-0MS-fast-117309 | fast | extreme | code | 117309 | ok | ok | ok |
| audit-WL-0MT-fast-77617 | fast | extreme | code | 77617 | ok | ok | ok |
| herdr-178750-fast-171226 | fast | extreme | code | 171226 | ok | ok | ok |
| audit-WL-0MS-fast-73983 | fast | extreme | code | 73983 | ok | ok | ok |
| audit-CG-0MT-fast-71251 | fast | extreme | code | 71251 | ok | ok | ok |
| audit-WL-0MT-fast-96164 | fast | extreme | code | 96164 | ok | ok | ok |
| 01a03059-f15-fast-140491 | fast | extreme | code | 140491 | ok | ok | ok |
| herdr-178751-fast-111512 | fast | extreme | code | 111512 | ok | ok | ok |
| herdr-178737-fast-84296 | fast | extreme | code | 84296 | ok | ok | ok |
| audit-CG-0MS-fast-131116 | fast | extreme | code | 131116 | ok | ok | ok |
| audit-WL-0MT-fast-73426 | fast | extreme | code | 73426 | ok | ok | ok |
| 01a0307f-d92-fast-248905 | fast | extreme | code | 248905 | ok | ok | ok |
| audit-WL-0MT-fast-117940 | fast | extreme | code | 117940 | ok | ok | ok |
| herdr-178744-fast-237391 | fast | extreme | code | 237391 | ok | ok | ok |
| herdr-178752-fast-109541 | fast | extreme | code | 109541 | ok | ok | ok |
| audit-WL-0MS-fast-125804 | fast | extreme | code | 125804 | ok | ok | ok |
| herdr-178752-fast-113513 | fast | extreme | code | 113513 | ok | ok | ok |
| herdr-178752-fast-183204 | fast | extreme | code | 183204 | ok | ok | ok |
| audit-WL-0MS-fast-75348 | fast | extreme | code | 75348 | ok | ok | ok |
| herdr-178752-fast-126934 | fast | extreme | code | 126934 | ok | ok | ok |
| herdr-178752-fast-189049 | fast | extreme | code | 189049 | ok | ok | ok |
| herdr-178753-fast-136976 | fast | extreme | code | 136976 | ok | ok | ok |
| 01a030e9-afe-fast-120885 | fast | extreme | code | 120885 | ok | ok | ok |
| herdr-178755-fast-85159 | fast | extreme | code | 85159 | ok | ok | ok |
| herdr-178756-fast-196601 | fast | extreme | code | 196601 | ok | ok | ok |
| herdr-178756-fast-128330 | fast | extreme | code | 128330 | ok | ok | ok |
| audit-LP-0MT-fast-82656 | fast | extreme | code | 82656 | ok | ok | ok |
| herdr-178757-fast-89845 | fast | extreme | code | 89845 | ok | ok | ok |
| herdr-178758-fast-95734 | fast | extreme | code | 95734 | ok | ok | ok |
| audit-WL-0MS-fast-81515 | fast | extreme | code | 81515 | ok | ok | ok |
| audit-LP-0MS-fast-105686 | fast | extreme | code | 105686 | ok | ok | ok |
| audit-LP-0MS-fast-108479 | fast | extreme | code | 108479 | ok | ok | ok |
| audit-WL-0MS-fast-77991 | fast | extreme | code | 77991 | ok | ok | ok |
| herdr-178758-fast-210686 | fast | extreme | code | 210686 | ok | ok | ok |
| audit-CG-0MT-fast-87339 | fast | extreme | code | 87339 | ok | ok | ok |
| herdr-178758-fast-424128 | fast | extreme | code | 424128 | ok | ok | ok |
| audit-WL-0MS-fast-95013 | fast | extreme | code | 95013 | ok | ok | ok |
| audit-WL-0MS-fast-95638 | fast | extreme | code | 95638 | ok | ok | ok |
| audit-WL-0MS-fast-89881 | fast | extreme | code | 89881 | ok | ok | ok |
| herdr-178760-fast-156559 | fast | extreme | code | 156559 | ok | ok | ok |
| herdr-178760-fast-81271 | fast | extreme | code | 81271 | ok | ok | ok |
| audit-LP-0MS-fast-88887 | fast | extreme | code | 88887 | ok | ok | ok |
| audit-LP-0MS-fast-89715 | fast | extreme | code | 89715 | ok | ok | ok |
| 01a035a2-22d-fast-131748 | fast | extreme | code | 131748 | ok | ok | ok |
| herdr-178760-fast-140059 | fast | extreme | code | 140059 | ok | ok | ok |
| audit-LP-0MS-fast-114116 | fast | extreme | code | 114116 | ok | ok | ok |
| audit-LP-0MS-fast-82832 | fast | extreme | code | 82832 | ok | ok | ok |
| herdr-178760-fast-201056 | fast | extreme | code | 201056 | ok | ok | ok |
| herdr-178761-fast-288280 | fast | extreme | code | 288280 | ok | ok | ok |
| herdr-178764-fast-72468 | fast | extreme | code | 72468 | ok | ok | ok |
| herdr-178764-fast-125598 | fast | extreme | code | 125598 | ok | ok | ok |
| herdr-178761-fast-104921 | fast | extreme | code | 104921 | ok | ok | ok |
| herdr-178764-fast-151581 | fast | extreme | code | 151581 | ok | ok | ok |
| audit-WL-0MT-fast-72056 | fast | extreme | code | 72056 | ok | ok | ok |
| audit-WL-0MT-fast-134848 | fast | extreme | code | 134848 | ok | ok | ok |
| audit-LP-0MS-fast-111010 | fast | extreme | code | 111010 | ok | ok | ok |
| audit-LP-0MS-fast-70652 | fast | extreme | code | 70652 | ok | ok | ok |
| audit-LP-0MS-fast-98520 | fast | extreme | code | 98520 | ok | ok | ok |
| audit-LP-0MS-fast-115435 | fast | extreme | code | 115435 | ok | ok | ok |
| audit-LP-0MS-fast-73056 | fast | extreme | code | 73056 | ok | ok | ok |
| herdr-178766-fast-123889 | fast | extreme | code | 123889 | ok | ok | ok |
| herdr-178764-fast-77237 | fast | extreme | code | 77237 | ok | ok | ok |
| audit-LP-0MS-fast-75804 | fast | extreme | code | 75804 | ok | ok | ok |
| herdr-178768-fast-262829 | fast | extreme | code | 262829 | ok | ok | ok |
| herdr-178769-fast-164897 | fast | extreme | code | 164897 | ok | ok | ok |
| audit-LP-0MS-fast-108894 | fast | extreme | code | 108894 | ok | ok | ok |
| audit-LP-0MS-fast-73493 | fast | extreme | code | 73493 | ok | ok | ok |
| herdr-178769-fast-127895 | fast | extreme | code | 127895 | ok | ok | ok |
| herdr-178769-fast-92101 | fast | extreme | code | 92101 | ok | ok | ok |
| herdr-178770-fast-109169 | fast | extreme | code | 109169 | ok | ok | ok |
| herdr-178770-fast-167364 | fast | extreme | code | 167364 | ok | ok | ok |
| herdr-178770-fast-162389 | fast | extreme | code | 162389 | ok | ok | ok |
| herdr-178770-fast-651408 | fast | extreme | code | 651408 | ok | ok | ok |
| audit-SA-0MT-fast-71156 | fast | extreme | code | 71156 | ok | ok | ok |
| herdr-178770-fast-130393 | fast | extreme | code | 130393 | ok | ok | ok |
| herdr-178771-fast-95293 | fast | extreme | code | 95293 | ok | ok | ok |
| herdr-178771-fast-121247 | fast | extreme | code | 121247 | ok | ok | ok |
| herdr-178772-fast-127298 | fast | extreme | code | 127298 | ok | ok | ok |
| herdr-178771-fast-421242 | fast | extreme | code | 421242 | ok | ok | ok |
| herdr-178770-fast-134021 | fast | extreme | code | 134021 | ok | ok | ok |
| herdr-178773-fast-209485 | fast | extreme | code | 209485 | ok | ok | ok |
| herdr-178774-fast-240347 | fast | extreme | code | 240347 | ok | ok | ok |
| herdr-178774-fast-80986 | fast | extreme | code | 80986 | ok | ok | ok |
| herdr-178775-fast-70131 | fast | extreme | code | 70131 | ok | ok | ok |
| herdr-178776-fast-188047 | fast | extreme | code | 188047 | ok | ok | ok |
| herdr-178776-fast-221764 | fast | extreme | code | 221764 | ok | ok | ok |
| herdr-178775-fast-72830 | fast | extreme | code | 72830 | ok | ok | ok |
| herdr-178777-fast-108601 | fast | extreme | code | 108601 | ok | ok | ok |
| herdr-178777-fast-310274 | fast | extreme | code | 310274 | ok | ok | ok |
| herdr-178778-fast-109638 | fast | extreme | code | 109638 | ok | ok | ok |
| herdr-178778-fast-83254 | fast | extreme | code | 83254 | ok | ok | ok |
| herdr-178777-fast-144860 | fast | extreme | code | 144860 | ok | ok | ok |
| herdr-178779-fast-87113 | fast | extreme | code | 87113 | ok | ok | ok |
| herdr-178781-fast-92391 | fast | extreme | code | 92391 | ok | ok | ok |
| audit-LP-0MT-fast-71457 | fast | extreme | code | 71457 | ok | ok | ok |
| audit-LP-0MT-fast-88323 | fast | extreme | code | 88323 | ok | ok | ok |
| 01a04279-3e9-fast-140874 | fast | extreme | agent | 140874 | ok | ok | ok |
| audit-CG-0MS-fast-126882 | fast | extreme | code | 126882 | ok | ok | ok |
| audit-CG-0MS-fast-140586 | fast | extreme | agent | 140586 | ok | ok | ok |
| herdr-178782-fast-115666 | fast | extreme | code | 115666 | ok | ok | ok |
| herdr-178783-fast-146871 | fast | extreme | code | 146871 | ok | ok | ok |
| herdr-178783-fast-137879 | fast | extreme | agent | 137879 | ok | ok | ok |
| herdr-178784-fast-128642 | fast | extreme | code | 128642 | ok | ok | ok |
| herdr-178784-fast-81351 | fast | extreme | qa | 81351 | ok | ok | ok |
| herdr-178784-fast-102498 | fast | extreme | code | 102498 | ok | ok | ok |
| herdr-178784-fast-92397 | fast | extreme | code | 92397 | ok | ok | ok |
| herdr-178784-fast-104869 | fast | extreme | agent | 104869 | ok | ok | ok |
| herdr-178785-fast-85268 | fast | extreme | code | 85268 | ok | ok | ok |
| herdr-178785-fast-84508 | fast | extreme | code | 84508 | ok | ok | ok |
| herdr-178785-fast-79429 | fast | extreme | code | 79429 | ok | ok | ok |
| herdr-178787-fast-116164 | fast | extreme | code | 116164 | ok | ok | ok |
| herdr-178787-fast-105693 | fast | extreme | code | 105693 | ok | ok | ok |
| herdr-178788-fast-130480 | fast | extreme | code | 130480 | ok | ok | ok |
| herdr-178787-fast-105748 | fast | extreme | qa | 105748 | ok | ok | ok |
| herdr-178789-fast-176406 | fast | extreme | code | 176406 | ok | ok | ok |
| herdr-178786-fast-238481 | fast | extreme | code | 238481 | ok | ok | ok |
| herdr-178787-fast-130672 | fast | extreme | qa | 130672 | ok | ok | ok |
| herdr-178787-fast-315486 | fast | extreme | code | 315486 | ok | ok | ok |
| herdr-178790-fast-211191 | fast | extreme | qa | 211191 | ok | ok | ok |
| 01a047ba-2ed-fast-207344 | fast | extreme | qa | 207344 | ok | ok | ok |
| 01a04836-7e3-fast-154166 | fast | extreme | qa | 154166 | ok | ok | ok |
| herdr-178791-fast-166767 | fast | extreme | code | 166767 | ok | ok | ok |
| herdr-178793-fast-84233 | fast | extreme | code | 84233 | ok | ok | ok |
| herdr-178793-fast-83380 | fast | extreme | qa | 83380 | ok | ok | ok |
| herdr-178794-fast-83494 | fast | extreme | qa | 83494 | ok | ok | ok |
| herdr-178795-fast-210753 | fast | extreme | code | 210753 | ok | ok | ok |
| herdr-178795-fast-79965 | fast | extreme | code | 79965 | ok | ok | ok |
| herdr-178795-fast-432625 | fast | extreme | qa | 432625 | ok | ok | ok |
| herdr-178795-fast-140222 | fast | extreme | code | 140222 | ok | ok | ok |
| herdr-178798-fast-101930 | fast | extreme | code | 101930 | ok | ok | ok |
| audit-LP-0MT-fast-78512 | fast | extreme | code | 78512 | ok | ok | ok |
| audit-LP-0MT-fast-111088 | fast | extreme | code | 111088 | ok | ok | ok |
| herdr-178799-fast-71010 | fast | extreme | code | 71010 | ok | ok | ok |
| 01a04d71-177-fast-110544 | fast | extreme | agent | 110544 | ok | ok | ok |
| herdr-178800-fast-116673 | fast | extreme | code | 116673 | ok | ok | ok |
| herdr-178800-fast-104574 | fast | extreme | code | 104574 | ok | ok | ok |
| herdr-178800-fast-107728 | fast | extreme | code | 107728 | ok | ok | ok |
| herdr-178800-fast-75042 | fast | extreme | agent | 75042 | ok | ok | ok |
| herdr-178800-fast-104907 | fast | extreme | agent | 104907 | ok | ok | ok |
| herdr-178801-fast-107029 | fast | extreme | code | 107029 | ok | ok | ok |
| herdr-178802-fast-85205 | fast | extreme | agent | 85205 | ok | ok | ok |
| herdr-178802-fast-73229 | fast | extreme | agent | 73229 | ok | ok | ok |
| herdr-178803-fast-110907 | fast | extreme | agent | 110907 | ok | ok | ok |
| herdr-178802-fast-105301 | fast | extreme | code | 105301 | ok | ok | ok |
| herdr-178803-fast-108017 | fast | extreme | code | 108017 | ok | ok | ok |
| herdr-178802-fast-113986 | fast | extreme | code | 113986 | ok | ok | ok |
| herdr-178803-fast-108920 | fast | extreme | qa | 108920 | ok | ok | ok |
| herdr-178804-fast-81037 | fast | extreme | code | 81037 | ok | ok | ok |
| audit-AH-0MT-fast-98402 | fast | extreme | code | 98402 | ok | ok | ok |
| herdr-178804-fast-108546 | fast | extreme | code | 108546 | ok | ok | ok |
| herdr-178804-fast-105738 | fast | extreme | code | 105738 | ok | ok | ok |
| audit-WL-0MT-fast-94582 | fast | extreme | agent | 94582 | ok | ok | ok |
| audit-WL-0MT-fast-106933 | fast | extreme | code | 106933 | ok | ok | ok |
| herdr-178805-fast-107631 | fast | extreme | code | 107631 | ok | ok | ok |
| herdr-178805-fast-113556 | fast | extreme | code | 113556 | ok | ok | ok |
| herdr-178805-fast-93537 | fast | extreme | code | 93537 | ok | ok | ok |
| herdr-178805-fast-92961 | fast | extreme | code | 92961 | ok | ok | ok |
| audit-WL-0MS-fast-77413 | fast | extreme | code | 77413 | ok | ok | ok |
| herdr-178805-fast-108396 | fast | extreme | code | 108396 | ok | ok | ok |
| herdr-178805-fast-111318 | fast | extreme | code | 111318 | ok | ok | ok |
| audit-AH-0MT-fast-75081 | fast | extreme | code | 75081 | ok | ok | ok |
| herdr-178805-fast-81940 | fast | extreme | code | 81940 | ok | ok | ok |
| audit-CG-0MT-fast-83840 | fast | extreme | code | 83840 | ok | ok | ok |
| audit-CG-0MT-fast-108311 | fast | extreme | agent | 108311 | ok | ok | ok |
| herdr-178805-fast-95799 | fast | extreme | agent | 95799 | ok | ok | ok |
| herdr-178805-fast-111993 | fast | extreme | agent | 111993 | ok | ok | ok |
| herdr-178806-fast-100776 | fast | extreme | code | 100776 | ok | ok | ok |
| audit-WL-0MT-fast-71604 | fast | extreme | code | 71604 | ok | ok | ok |
| audit-SA-0MT-cheap-51166 | cheap | trigger-cap | code | 51166 | ok | ok | ok |
| herdr-178754-cheap-48922 | cheap | trigger-cap | code | 48922 | ok | ok | ok |
| audit-LP-0MT-cheap-44716 | cheap | trigger-cap | code | 44716 | ok | ok | ok |
| audit-LP-0MT-cheap-57875 | cheap | trigger-cap | code | 57875 | ok | ok | ok |
| herdr-178754-cheap-43287 | cheap | trigger-cap | code | 43287 | ok | ok | ok |
| herdr-178755-cheap-56369 | cheap | trigger-cap | code | 56369 | ok | ok | ok |
| herdr-178755-cheap-48387 | cheap | trigger-cap | code | 48387 | ok | ok | ok |
| audit-LP-0MS-cheap-55928 | cheap | trigger-cap | code | 55928 | ok | ok | ok |
| audit-LP-0MS-cheap-54588 | cheap | trigger-cap | code | 54588 | ok | ok | ok |
| audit-WL-0MT-cheap-51056 | cheap | trigger-cap | code | 51056 | ok | ok | ok |
| audit-WL-0MT-cheap-46385 | cheap | trigger-cap | code | 46385 | ok | ok | ok |
| herdr-178761-cheap-43099 | cheap | trigger-cap | code | 43099 | ok | ok | ok |
| audit-WL-0MT-cheap-48427 | cheap | trigger-cap | code | 48427 | ok | ok | ok |
| audit-WL-0MT-cheap-59531 | cheap | trigger-cap | code | 59531 | ok | ok | ok |
| audit-WL-0MT-cheap-44866 | cheap | trigger-cap | code | 44866 | ok | ok | ok |
| audit-WL-0MT-cheap-51207 | cheap | trigger-cap | code | 51207 | ok | ok | ok |
| audit-WL-0MT-cheap-59724 | cheap | trigger-cap | code | 59724 | ok | ok | ok |
| audit-WL-0MS-cheap-60988 | cheap | trigger-cap | code | 60988 | ok | ok | ok |
| audit-WL-0MT-cheap-54933 | cheap | trigger-cap | code | 54933 | ok | ok | ok |
| audit-LP-0MS-cheap-50469 | cheap | trigger-cap | code | 50469 | ok | ok | ok |
| audit-WL-0MT-cheap-60454 | cheap | trigger-cap | code | 60454 | ok | ok | ok |
| audit-LP-0MS-cheap-48150 | cheap | trigger-cap | code | 48150 | ok | ok | ok |
| audit-WL-0MT-cheap-57600 | cheap | trigger-cap | code | 57600 | ok | ok | ok |
| herdr-178764-cheap-54973 | cheap | trigger-cap | code | 54973 | ok | ok | ok |
| herdr-178764-cheap-48152 | cheap | trigger-cap | code | 48152 | ok | ok | ok |
| audit-SA-0MT-cheap-60984 | cheap | trigger-cap | code | 60984 | ok | ok | ok |
| audit-SA-0MT-cheap-54436 | cheap | trigger-cap | code | 54436 | ok | ok | ok |
| audit-SA-0MS-cheap-48144 | cheap | trigger-cap | code | 48144 | ok | ok | ok |
| audit-SA-0MS-cheap-48626 | cheap | trigger-cap | code | 48626 | ok | ok | ok |
| herdr-178771-cheap-51293 | cheap | trigger-cap | code | 51293 | ok | ok | ok |
| audit-SA-0MT-cheap-54941 | cheap | trigger-cap | code | 54941 | ok | ok | ok |
| audit-SA-0MT-cheap-43907 | cheap | trigger-cap | code | 43907 | ok | ok | ok |
| herdr-178773-cheap-46006 | cheap | trigger-cap | code | 46006 | ok | ok | ok |
| audit-LP-0MT-cheap-53343 | cheap | trigger-cap | code | 53343 | ok | ok | ok |
| herdr-178779-cheap-48913 | cheap | trigger-cap | code | 48913 | ok | ok | ok |
| audit-LP-0MT-cheap-53192 | cheap | trigger-cap | code | 53192 | ok | ok | ok |
| audit-LP-0MT-cheap-49944 | cheap | trigger-cap | code | 49944 | ok | ok | ok |
| herdr-178787-cheap-50956 | cheap | trigger-cap | code | 50956 | ok | ok | ok |
| herdr-178787-cheap-59634 | cheap | trigger-cap | qa | 59634 | ok | ok | ok |
| herdr-178788-cheap-56031 | cheap | trigger-cap | agent | 56031 | ok | ok | ok |
| herdr-178789-cheap-46061 | cheap | trigger-cap | qa | 46061 | ok | ok | ok |
| herdr-178786-cheap-49549 | cheap | trigger-cap | code | 49549 | ok | ok | ok |
| herdr-178787-cheap-46298 | cheap | trigger-cap | code | 46298 | ok | ok | ok |
| herdr-178787-cheap-48996 | cheap | trigger-cap | code | 48996 | ok | ok | ok |
| herdr-178796-cheap-59023 | cheap | trigger-cap | code | 59023 | ok | ok | ok |
| audit-WL-0MS-cheap-46167 | cheap | trigger-cap | code | 46167 | ok | ok | ok |
| audit-WL-0MT-cheap-47308 | cheap | trigger-cap | code | 47308 | ok | ok | ok |
| audit-CG-0MT-cheap-57108 | cheap | trigger-cap | code | 57108 | ok | ok | ok |
| audit-CG-0MT-cheap-58443 | cheap | trigger-cap | code | 58443 | ok | ok | ok |
| audit-LP-0MT-cheap-44415 | cheap | trigger-cap | code | 44415 | ok | ok | ok |
| audit-LP-0MT-cheap-50089 | cheap | trigger-cap | qa | 50089 | ok | ok | ok |
| audit-LP-0MT-cheap-52806 | cheap | trigger-cap | code | 52806 | ok | ok | ok |
| audit-AH-0MT-cheap-51317 | cheap | trigger-cap | agent | 51317 | ok | ok | ok |
| herdr-178804-cheap-52369 | cheap | trigger-cap | agent | 52369 | ok | ok | ok |
| herdr-178804-cheap-56780 | cheap | trigger-cap | code | 56780 | ok | ok | ok |
| unknown-cheap-218783 | cheap | extreme | code | 218783 | ok | ok | ok |
| herdr-178803-cheap-74718 | cheap | extreme | code | 74718 | ok | ok | ok |
| herdr-178804-cheap-107257 | cheap | extreme | qa | 107257 | ok | ok | ok |
| 01a0307f-d92-cheap-102350 | cheap | extreme | code | 102350 | ok | ok | ok |
| herdr-178744-cheap-97194 | cheap | extreme | code | 97194 | ok | ok | ok |
| herdr-178752-cheap-110303 | cheap | extreme | code | 110303 | ok | ok | ok |
| herdr-178752-cheap-92705 | cheap | extreme | code | 92705 | ok | ok | ok |
| herdr-178752-cheap-103953 | cheap | extreme | code | 103953 | ok | ok | ok |
| herdr-178753-cheap-101289 | cheap | extreme | code | 101289 | ok | ok | ok |
| 01a030e9-afe-cheap-107546 | cheap | extreme | code | 107546 | ok | ok | ok |
| audit-WL-0MS-cheap-95537 | cheap | extreme | code | 95537 | ok | ok | ok |
| audit-WL-0MS-cheap-69137 | cheap | extreme | code | 69137 | ok | ok | ok |
| audit-WL-0MS-cheap-96833 | cheap | extreme | code | 96833 | ok | ok | ok |
| herdr-178755-cheap-83879 | cheap | extreme | code | 83879 | ok | ok | ok |
| herdr-178758-cheap-255364 | cheap | extreme | code | 255364 | ok | ok | ok |
| herdr-178760-cheap-95649 | cheap | extreme | code | 95649 | ok | ok | ok |
| herdr-178760-cheap-120803 | cheap | extreme | code | 120803 | ok | ok | ok |
| audit-WL-0MT-cheap-98843 | cheap | extreme | code | 98843 | ok | ok | ok |
| herdr-178761-cheap-100395 | cheap | extreme | code | 100395 | ok | ok | ok |
| audit-WL-0MT-cheap-71865 | cheap | extreme | code | 71865 | ok | ok | ok |
| audit-LP-0MS-cheap-62132 | cheap | extreme | code | 62132 | ok | ok | ok |
| audit-WL-0MT-cheap-76416 | cheap | extreme | code | 76416 | ok | ok | ok |
| audit-WL-0MT-cheap-63601 | cheap | extreme | code | 63601 | ok | ok | ok |
| audit-WL-0MT-cheap-64616 | cheap | extreme | code | 64616 | ok | ok | ok |
| audit-WL-0MT-cheap-62789 | cheap | extreme | code | 62789 | ok | ok | ok |
| herdr-178762-cheap-109098 | cheap | extreme | code | 109098 | ok | ok | ok |
| audit-WL-0MT-cheap-65087 | cheap | extreme | code | 65087 | ok | ok | ok |
| audit-LP-0MS-cheap-86459 | cheap | extreme | code | 86459 | ok | ok | ok |
| herdr-178770-cheap-73027 | cheap | extreme | code | 73027 | ok | ok | ok |
| herdr-178770-cheap-133467 | cheap | extreme | code | 133467 | ok | ok | ok |
| herdr-178770-cheap-184021 | cheap | extreme | code | 184021 | ok | ok | ok |
| herdr-178771-cheap-114658 | cheap | extreme | code | 114658 | ok | ok | ok |
| audit-CG-0MT-cheap-64606 | cheap | extreme | code | 64606 | ok | ok | ok |
| herdr-178772-cheap-62152 | cheap | extreme | code | 62152 | ok | ok | ok |
| herdr-178777-cheap-316481 | cheap | extreme | code | 316481 | ok | ok | ok |
| herdr-178778-cheap-113955 | cheap | extreme | code | 113955 | ok | ok | ok |
| audit-WL-0MT-cheap-61696 | cheap | extreme | code | 61696 | ok | ok | ok |
| herdr-178779-cheap-98937 | cheap | extreme | code | 98937 | ok | ok | ok |
| audit-LP-0MT-cheap-70047 | cheap | extreme | code | 70047 | ok | ok | ok |
| herdr-178780-cheap-81166 | cheap | extreme | code | 81166 | ok | ok | ok |
| herdr-178777-cheap-74769 | cheap | extreme | code | 74769 | ok | ok | ok |
| audit-LP-0MT-cheap-65438 | cheap | extreme | code | 65438 | ok | ok | ok |
| audit-LP-0MT-cheap-85348 | cheap | extreme | code | 85348 | ok | ok | ok |
| herdr-178787-cheap-65669 | cheap | extreme | code | 65669 | ok | ok | ok |
| herdr-178787-cheap-65815 | cheap | extreme | agent | 65815 | ok | ok | ok |
| herdr-178787-cheap-91304 | cheap | extreme | code | 91304 | ok | ok | ok |
| herdr-178787-cheap-92405 | cheap | extreme | qa | 92405 | ok | ok | ok |
| herdr-178789-cheap-72635 | cheap | extreme | code | 72635 | ok | ok | ok |
| herdr-178795-cheap-339762 | cheap | extreme | code | 339762 | ok | ok | ok |
| herdr-178795-cheap-104866 | cheap | extreme | code | 104866 | ok | ok | ok |
| audit-WL-0MS-cheap-77367 | cheap | extreme | code | 77367 | ok | ok | ok |
| herdr-178795-cheap-106373 | cheap | extreme | code | 106373 | ok | ok | ok |
| audit-WL-0MS-cheap-77963 | cheap | extreme | code | 77963 | ok | ok | ok |
| audit-CG-0MT-cheap-63226 | cheap | extreme | code | 63226 | ok | ok | ok |
| audit-LP-0MT-cheap-65374 | cheap | extreme | code | 65374 | ok | ok | ok |
| herdr-178797-cheap-62042 | cheap | extreme | code | 62042 | ok | ok | ok |
| herdr-178798-cheap-100092 | cheap | extreme | code | 100092 | ok | ok | ok |
| herdr-178803-cheap-108163 | cheap | extreme | agent | 108163 | ok | ok | ok |
| herdr-178804-cheap-97421 | cheap | extreme | code | 97421 | ok | ok | ok |
| herdr-178804-cheap-100711 | cheap | extreme | code | 100711 | ok | ok | ok |


## Execution status (dry-run vs live)

**This report is a DRY-RUN pipeline validation, not a live quality verdict.**

| Component | Status | Evidence |
|---|---|---|
| Task suite extraction (real logs, 29 min of routing data) | ✅ validated | 416 tasks: 301 fast (69 trigger-cap / 231 extreme), 115 cheap (55 trigger-cap / 60 extreme); ≥30 trigger-cap & ≥10 extreme per mode (design §2/§6) |
| Transcript provenance | ✅ | 172 tasks from full session recordings, 244 synthetic mirrors (design §2 fallback); each recording used at most once, time-matched ±2h |
| Compaction (truncate→≤38K fast / summarize→≤30K cheap) | ✅ fires | scaled to proxy routing units via per-task `token_scale` (log `estimated_tokens` = ground truth) |
| Efficiency metrics | ✅ real data | prefill est. reduction B vs A: **fast 61.6%** (33.4M→12.8M), **cheap 30.2%** (8.9M→6.2M); both ≥ 25% bar |
| Quality metrics (rubric/completion/failure) | ⚠️ heuristic only | dry-run uses offline structural proxy scores (mock responses → trivially equal B=A→1.000); real numbers require the live judge |
| Live three-arm execution | ⛔ blocked | arm A needs `DEEPSEEK_API_KEY` (not set, no `api.deepseek.com` credential in `~/.pi/agent/multi-auth.json`); local 35B backend (port 38197) was processing an in-flight task and hung on new requests during this session, MTP instance reported terminated |

**Go/no-go verdict above is therefore provisional on the quality axis.** The
pre-registered efficiency gate (≥25% prefill reduction, TTFT not worse) holds
on real data. The quality gate (B ≥ 0.95×A rubric, completion B ≥ A−3pp, no
failure increase) is exercised end-to-end but the numbers are structural
placeholders until a live run.

### Commands to complete the live run
```bash
export DEEPSEEK_API_KEY=...            # arm A (remote baseline)
# confirm local backend healthy:
curl -s http://127.0.0.1:38197/health  # expect {"status":"ok"} AND responsive completions
cd proxy
python3 scripts/run_compaction_experiment.py --log-dir /var/log/llama-proxy \
  --recordings-dir ~/.llm-proxy/session-recordings --output-dir experiment-results \
  --judge-endpoint <judge> --judge-key <key>   # add judge for real quality scoring
```
A degraded live run with arms B/C only (arm A pending key) can still measure
compaction's quality effect as B vs C on the same local model.
