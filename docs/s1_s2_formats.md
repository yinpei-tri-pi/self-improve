# The current S1 and S2 training data formats (as of 2026-11-23)

Reviewed from the code that built and reads the data: RoboAnnotator, openpi (branch robocasa), sys2_train_eval, and the eval
loops in openpi `examples/robocasa` and sys1_sys2. Short path forms used below:

| short form | path |
|---|---|
| OP | `~/openpi/src/openpi` |
| RA | `~/RoboAnnotator` |
| HX | `~/HexaShell-s1/hexashell/benches/robocasa` |

## S1: pi0.5 subgoal-conditioned VLA (`pi05_robocasa_system1`, flags `--data.progress-as-action --model.no-use-progress-head --data.no-include-executed-step`)

**Base set** (`system1_fullset_0717`):
- 30,897 demo episodes and 34.2M samples (one per frame), in 16,736 shards of 2,048 samples (about 2.15 TB).
- By variant: normal 28.66M; incomplete "continue to X" 4.92M; failure-augmented recoveries 0.6M.
- Quality: "Success" 33.24M frames, "Suboptimal" 0.96M.
- Segments are the spans of Gemini's dense fine-step annotation. Each frame in a span is one sample, plus one halt frame after the span.

**Inputs for one sample at frame t of a subgoal span [s, e]:**
- **Images:** 6 at 224 px (stored as 256 px JPEG). These are the 3 cameras (agentview left, agentview right, eye-in-hand) at frame t, and the same 3 at the anchor (the span's first frame). A learned role embedding marks current vs anchor.
- **State:** the lean 14-d state of the current frame and of the anchor, 28 values in all. The 14 dims are:
  - eef pos (3) and eef rot6d (6);
  - gripper width (1);
  - base xy relative to the episode start (2);
  - sin/cos of the base yaw relative to the start (2).

  Values are quantile-normalized, cut into 256 bins and written as integers in the prompt; pi0.5 has no continuous state input. The raw 16-d state is also shipped, so the policy can recompute the lean state and assert that it matches.
- **Prompt (example):**

  ```
  Task: open the right drawer; Current Subgoal: grasp the drawer handle
  Quality: Success; Estimated Length: 50
  Initial State: 99 24 31 ... 128 255; Current State: 118 15 11 ... 128 255; Current Gripper: Close;
  Action:
  ```
  The fields:
  - `Task`: the episode goal, lowercased.
  - `Subgoal`: the Gemini fine-step text.
  - `Quality`: Success, or Suboptimal for merged failure+recovery spans and the slowest 1% of manipulation spans.
  - `Estimated Length`: the span length snapped to the grid [50, 75, ..., 1300].
  - `Current Gripper`: the sign of the previous gripper command.
  - `Executed Step` is still in the data but dropped by the flag.

**Output:**
- A chunk of 20 actions × 11 lean dims, normalized. The 11 dims are base vx/vy/yaw (3), mode (±1), eef Δpos (3), eef Δrot (3) and gripper (±1); torso is dropped because it is always 0.
- Steps past the span end are a settle pad: velocities 0, mode and gripper held.
- Progress as action: one extra dim holding 2p − 1, where p ramps linearly from (t − s)/(e − s) to 1 at the span end. The halt frame is all 1.
- Everything is padded to 32 dims. The flow-matching loss averages over all 32, the 20 zero-pad dims included.

**At inference:**
- The subgoal comes from S2 (or Astra in HexaShell), the task from the env, and Quality is always "Success".
- Estimated Length comes from S2's `<estimated_step>` in combined_eval, or from the per-subgoal training-mode table in HexaShell (`s1_est.json`).
- The anchor is the observation when the call starts.
- HexaShell runs 16 of the 20 actions per query and stops at progress ≥ 0.95 with the arm still.

**Carried but unused in production:**
- Arrays: `last_action*`, the raw 12-d chunks, the episode-padded chunks and their masks (left from a pad ablation). The baked lean states are kept only for the assert.
- Meta: `executed_step`, a stale `prompt` placeholder, `subgoal_detail`, the `milestone_*` fields, the 10-way progress classes (left from the progress head), the anchor lookup keys (`anchor_key`, `anchors_dir`), and the leak and fail fields (`lead_frames`, `leak_*`, `fail_end`).
- Quirks:
  - The loader still applies a rise-drop-recover progress overlay to 2,983 Suboptimal spans, although the writer reverted to a linear ramp.
  - The incomplete "continue to X" spans repeat frames of normal spans under another subgoal text.
- Self-improvement fields (`si_*`, only in our rollout shards): only `si_label` and `si_weight` are read, by the uncommitted openpi-si objective.

## S2: Qwen3-VL-4B planner (`system2-full-0804`, checkpoint-17124)

**Training set:**
- 30,931 demo episodes on 299 pretrain tasks (plus 100 target episodes of CloseBlenderLid and GetToastedBread).
- 1.10M records: PLAN 62k, EXECUTION 640k, SUMMARY 397k.
- Each record is one [system, user, assistant] triple; no chat history.

**PLAN** (called once per episode):
- Input: the goal, plus one image of the first frame, the 3 cameras tiled to 256×768 (about 192 vision tokens).
- Output: `<thought>…</thought><plan>- [ ] M1: …</plan>`, a list of milestones.
- A "with memory" variant adds a recalled recipe, which in training is the same episode's own (hindsight).

**EXECUTION** (called every turn):
- **First-turn input:** the goal, the plan and the start image: "issue the first fine step".
- **Later-turn input:**
  - the goal and the current plan checklist (the only memory between turns);
  - a 4 fps video of the fine step just executed (at most 32 frames, 256×768);
  - privileged lines: `Current task status: ongoing|finished` and `Current gripper status: open|close|unsure`.
- **Output tags:**
  - `<thought>` and `<judge>`, one of task_begin, subgoal_complete, subgoal_incomplete, subgoal_failed or task_finish;
  - `<plan_update>` (milestone ids, merged into the plan);
  - `<estimated_step>`, on a grid from 50 to 1300;
  - `<subgoal>`, which becomes S1's prompt;
  - `<subgoal_detail>`.
- **Judge counts:** complete 447k, incomplete 95k, task_finish 36k, task_begin 31k, failed 31k.
- **Negatives:**
  - "incomplete": demo spans cut short.
  - "failed": failures injected in MuJoCo, followed by scripted recoveries, atomic tasks only, repeated 10×.

**SUMMARY** (36% of the data):
- Narrate a 4 s clip, then write a recipe from all the narrations.
- Never used by the default eval loop; only a memory variant uses it, privileged.

**What the eval loop actually uses:**
- The plan and `plan_update` (as memory), `subgoal`, and `estimated_step`. The estimate sets S1's budget, min(400, 2·est), and its length conditioning. The language-steering rules override it, resolving to the maximum proposal.
- `task_finish` ends the episode. It can never be right at inference, because the loop stops on env success first, so S2 always sees "ongoing". The final-milestone retry rule cancels it.
- `subgoal_failed` is read by 2 task rules. complete vs incomplete is never branched on.
- Not used at all: `thought`, `subgoal_detail`, SUMMARY.
- About 22 task rules plus general rules rewrite S2's output (repeat caps, regrasp recovery, estimate floors and pins, plan inserts).

**Train vs inference mismatches:**
- Clip frame sampling: training uses a linspace over the span; inference drops static frames and takes every 5th frame.
- Goal casing: lowercased first letter in training, raw at inference.
- The privileged status lines.

## Our rollouts (HexaShell, Astra + S1), what they hold for new data

Per turn (`events.jsonl`, `cells.jsonl`, `transcript.json`):
- Astra's thought and judge of the previous step, and the next subgoal.
- Who executes it (`by`: s1 or agent).
- For S1 runs: steps, stop reason (progress or budget), progress and est_length.
- For the model's own calls: the code and robot calls.

Per control step (`trajectory/`):
- 3 cameras at 256 px, S1's raw state, the 12-d action, the subgoal and who moved, and the full MuJoCo state.
- Once per episode: the scene XML and the episode meta.
- Episodes start from recorded demos (split `demo`).

`selfimprove/s2/build.py` drafts S2 EXECUTION-style rows from these turns: the same templates and 256×768 4 fps clips. Plan, plan_update and subgoal_detail are still missing.
