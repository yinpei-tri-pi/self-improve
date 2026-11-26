# Prompt A/B on the 108 selected tasks (2026-11-22)

Tables: `docs/ab108.md`, `docs/ab108r2.md`, `docs/ab108_ranking.md`. The raw episodes (`self_improve/ab108_{with,without}`, `ab108r2_*`)
were deleted on 2026-11-24 (recorded before the harness followed the demos' action convention, and not from recorded starts).

## Setup

- **Tasks.** The 108 tasks of rule A (`assets/selected_tasks.json`): 44 atomic and 64 composite. None is a target task.
- **Seeds and split.** 5 fresh seeds per task (`seed_key ab108`, RoboCasa pretrain split), the same seeds in both arms. They are not the pilot's seeds.
- **Models.** S1 is the 270k checkpoint. Astra is gpt-6.2-astra with high effort and `s1=first`.
- **Arms.**
  - **with:** the new task notes (`skills/robocasa/<Task>.md`) and System1 step tables (`skills/robocasa_s1/<Task>.md`). These were written from each task's code, its 2 pilot episodes and the target-task notes of the same family or fixture (18 parallel subagents).
  - **without:** the generic card (`skills=no s1_skills=no`), as in the pilot.
- **Cluster.** One rollout cluster in us-west-2 with 8 GPUs per arm. g6e capacity was short, so the autoscaler mixed g6e.2xlarge, 4xlarge and 8xlarge. Each arm took about 1 h: with 0.97 h, without 1.18 h.

## Result (540 paired episodes, no simulator or gateway errors)

| | with | without | diff | discordant pairs (with only / without only) | McNemar p |
|---|---|---|---|---|---|
| all | **88.1%** | 83.9% | +4.3 | 56 / 33 | 0.019 |
| atomic (220) | 89.1% | 88.6% | +0.5 | 15 / 14 | 1.0 |
| composite (320) | **87.5%** | 80.6% | **+6.9** | 41 / 19 | 0.006 |

- **The gain is all in the composites.** 12 of the 16 families gained; managing_freezer_space, organizing_recycling, making_toast and arranging_condiments gained +14 to +20 points. Atomic tasks were already near ceiling with the generic card.
- **System1 does more of the work with the prompts.** Its step share went from 0.68 to 0.74 on the composites. That is more S1-executed data per episode, at the same Astra tokens per episode (698k vs 693k prompt tokens).
- **Failure causes.** Classified from the model's own finish summary:

  | | dropped object (floor, out of reach) | out of steps | other |
  |---|---|---|---|
  | with (64 failures) | 32 | 32 | 0 |
  | without (87 failures) | 47 | 38 | 2 |

  Half of all failures are objects slipping out of the gripper during a lift, a carry or a base turn. The arm cannot grasp below about 5 cm, so a dropped object is lost. This is the largest remaining lever, and it is not task-specific. Options:
  - re-check the grip before every drive;
  - slower base acceleration and turn rate while holding;
  - re-grasp when the gap narrows;
  - turn in small legs.

## Recommendations for the self-improvement rollouts (to confirm)

- **Use the prompts** (`skills=yes`, `s1_skills=yes`, the default in `configs/default.yaml`) on all 108 tasks.
- **Keep 107 of 108** (`assets/selected_tasks.json` still lists all 108 until you confirm). Drop only **FreezeIceTray**:
  it was 2/5 in both arms (0/2 in the pilot), and its failures are the tray slipping during the drive to the freezer,
  which is not useful S1 data. ScalePortioning was also a drop candidate (0/5 with the first notes), but with the
  revised notes it went 4/6 on fresh seeds (below), so it stays.
- **Revised after the A/B:** the notes of the five tasks that did worse with the prompts. These are CloseOven 3/5 vs 5/5, SlideOvenRack 3/5 vs 5/5, MoveToFreezerDrawer 2/5 vs 4/5, ScalePortioning 0/5 vs 2/5 and BeverageOrganization 1/5 vs 4/5. The revisions come from a seed-by-seed comparison of the two arms.

  **Re-check on fresh seeds** (`ab108r2_{with,without}`: 6 new seeds per task, both arms; `docs/ab108r2.md`):

  | task | with (revised notes) | without |
  |---|---|---|
  | CloseOven | 3/6 | 3/6 |
  | SlideOvenRack | 5/6 | 6/6 |
  | MoveToFreezerDrawer | 5/6 | 5/6 |
  | ScalePortioning | 4/6 | 5/6 |
  | BeverageOrganization | 6/6 | 6/6 |
  | total | 23/30 | 25/30 |

  The regressions are gone: parity within noise, and no task is more than one episode apart. CloseOven's 450-step
  budget is tight in both arms.
- **Lowest of the kept tasks (with arm, 3/5):** CoffeeServeMug, PickPlaceCounterToOven, PreheatOven, TurnOffMicrowave, plus several 3/5 composites. Most of these ran out of steps on short horizons (300-600). These tasks stay: their successful episodes are clean data.
