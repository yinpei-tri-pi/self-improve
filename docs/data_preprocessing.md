# Data preprocessing: S1 + HexaShell rollouts -> System1 and System2 training data

Status 2026-11-25: the design as agreed in the data-processing debugging phase (implemented; round 0 of si_sub50
as the worked example). Code: `selfimprove/data/s1_select.py` (which spans), `selfimprove/data/s1_shards.py` (System1
shards), `selfimprove/s2/rows_v2.py` (System2 rows). The rollout sites show every rule per turn
(`scripts/build_debug_site.py`).

## Inputs: one rollout episode

The HexaShell record of an episode played by Astra (the planner) with System1 (`s1=first`) from a recorded dataset
start (split `demo`): the trajectory (every control step at 20 Hz: 3 cameras 256x256, System1's raw state, the 12-d
action in RoboCasa's convention, the gripper level, the MuJoCo state), the turns (`cells.jsonl`: thought, judge,
subgoal, milestone, milestones, planner_reasoning, code, output) and the result (success from the benchmark).

A *segment* is the control steps of one turn's motion, executed by System1 (`robot.s1()`) or by the harness (Astra's
own calls); the next turn's judge says how it went (complete, incomplete, failed).

## System1 training data

**Which episodes:** successful ones only.

**Span:** a step's control steps; prompt = the subgoal; anchor picture and state = the span's start; Estimated Length =
the length grid's value at or below its length, at least 50; Quality Success; progress 0 -> 1 over the span; one halt
frame after it (none after the episode's last frame or after a step left partway). Plain supervised objective.

| kind | what it is | kept when |
|---|---|---|
| `s1` System1's own | a System1 segment, its subgoal as given ("continue to X" / "X again" included) | the next turn judged it complete; dropped when judged incomplete or failed, or stopped at its step budget |
| `harness` | consecutive harness turns on one step (X, "continue to X", "X again"), in the words of the first turn | completed, or left partway (no halt frame); dropped when judged failed, very jerky (jerk over the demos' p99.9 on >10% of steps, or >3x it once), whole-body (base and arm together), or under 10 steps |
| `harness_merged` | consecutive kept harness spans each under 50 steps, joined until the span reaches 50 steps, worded as the last ("grasp X" = reach + grasp, "retract the arm" = release + retract; gripper changes allowed) | never a navigation with a manipulation, never across a milestone, never a recovery fix, not when the merged motion is very jerky |
| `recovery_within` | System1's miss at X (failed, incomplete or a timeout) + its fix: the harness's (<= 2 turns, miss + fix < 125 steps; no bound for a navigation) or System1's own retry of X (a harness preparation in its turn included; no bound), worded X, anchored at System1's start | after a failure or a timeout System1's frames are context only (anchor, progress) and only the fix's frames are trained; after an incomplete step all frames are |
| `recovery_request` | the fix alone ("X again" / "continue to X"): the harness's or System1's retry | as `harness` / `s1` |

A two-part turn - a few harness motions (opening the gripper, a small move), then `robot.s1()` in the same turn - is one
System1 step under the turn's subgoal, the preparation included.

Other rules: a gripper held partly open/closed by the harness is relabeled to the demos' -1 / +1 (the state keeps the
real finger opening); a manipulation over 400 steps is cut into X and "continue to X", a navigation stays whole; in a
run of harness steps with more than 2 manipulation steps and a miss, only the clean tail after the last miss is kept.

**Shards:** openpi's RoboCasa webdataset (the fields its loader reads; `si_kind`, `si_source`, `si_turns` as
provenance), checked through openpi's own loader with the checkpoint's norm stats. Training: 80% rollout spans, 20% the
base set, batch 256, 15k steps from the 270k checkpoint.

Round 0 (1,355 successful episodes): 16,573 spans, 928,383 samples; 413 merged fragments, 593 recoveries within the
step, 577 on request.

## System2 training data

**Which episodes:** all, failures included; **what:** one row per turn that moved the robot (short harness turns joined
as System1's fragments are: one row at the first, decision expert, the merged words), plus one closing row per episode.

**Input (user text):**

```
The task is: <the instruction>

Milestones:
  M1: ... (done)
  M2: ... (now, again)
    - <step> (done)
    (back from M3)
    - <step> (failed)
  M3: ... (to do, tried before)

The last step was '<subgoal>', executed by the model | the expert. Here is its video:
<video>
Current gripper: open | close | unsure.
Current task status: begin | ongoing | success | failed.
```

- Milestones: all of them always (the first turn's list, or Astra's latest); before the current one = done, after =
  to do. Only the current one shows steps: its last 5 (the attempts at one step - X, "continue to X", "X again" - are
  one line with how the latest went: done / partial / failed). Going back (M3 -> M2): M2 "now, again" with a
  "(back from M3)" marker, its earlier steps kept; M3 emptied, "to do, tried before".
- The video: the last step, 4 fps, the 3 cameras tiled 256x768, at most 32 frames; the first turn: the opening scene
  as one picture and "Current task status: begin".
- A row whose video has under 4 frames (a step of 13 control steps or fewer) is not trained; its step stays in the
  later rows' milestone view. One row per S1 step: a merged step is one row judging the merged video (decided 2026-11-25).
- The task status comes from the benchmark (privileged, as the old System2's): the evaluation must pass it the same way.

**Target (assistant text):** `<thought>` (Astra's planner_reasoning: System1 is "the model", the harness "the expert"),
`<judge>` (task_begin, subgoal_complete, subgoal_incomplete, subgoal_failed), `<milestones>` (first turn),
`<milestone>`, `<decision>` (model | expert: who executed the step; a turn that called System1 is model), `<subgoal>`.
Closing row: the last step's video and the status success -> `<judge>task_finish</judge>`, failed -> `task_failed`.

Round 0: 24,545 rows (1,289 merged steps); the milestone view 6 lines at the median, 16 at most.

**Shards (next):** the `system2-shards` webdataset that `sys2_train_eval`'s ms-swift plugin reads
(`<key>.json` with the messages, `<key>.videobin` or `<key>.imagebin0`, `manifest.jsonl` with num_samples per shard),
for Qwen3-VL-8B; all rounds' rows so far. On hold: the old full0804 PLAN + EXECUTION records converted to this format as
20% of the mix.

## Rollout-side choices that shape the data

- HexaShell (Astra) writes the planner fields every turn; planner_reasoning is training text for System2 ("the model",
  "the expert", the milestones' rationale at the start and at each switch).
- Astra's own turns: one step per turn, except a short reach joined to its grasp and a release to its retract.
- Gripper: a step ends only once the fingers have finished moving (System1's stop rule and the harness's grip).
- Split demo: the recorded start; RoboCasa's randomly placed fixtures synced to the loaded scene, the recording's own
  instruction; from round 1 the start perturbed up to 30 cm / 30 degrees.
