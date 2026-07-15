Debugging / comparing script changes:
  use the same seed, such as --seed 42

Exploring gait robustness:
  run several seeds, such as --seed 1, 2, 3, 4, 5

Final confidence check:
  train/evaluate across multiple seeds, not just one lucky run

A deterministic fresh run with a fixed seed that creates a video at each model checkpoint(1000 iterations by default adding   agent.save_interval=250   sets save checkpoint to every 250 iterations):

python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Humanoid-v0 \
  --num_envs 4096 \
  --max_iterations 15000 \
  --seed 42 \
  --run_name fresh_seed42 \
  --video \
  --video_length 300 \
  --headless 


Resume example: 

python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Humanoid-v0 \
  --num_envs 4096 \
  --max_iterations 1000 \
  --seed 42 \
  --resume \
  --load_run 2026-07-01_12-45-00_fresh_seed42 \
  --checkpoint model_15000.pt \
  --run_name resume_seed42_from_15000 \
  --video \
  --video_length 300 \
  --headless 


export example:

python scripts/rsl_rl/export_policy.py \
  --task Velocity-Lilgreen-Humanoid-v0 \
  --load_run 2026-07-01_20-33-02_resume_seed42_from_15000\
  --checkpoint model_15999.pt \
  --num_envs 1 \
  --headless

play joystic example:

python scripts/rsl_rl/play_joystick.py \
  --task Velocity-Lilgreen-Humanoid-v0 \
  --load_run 2026-07-01_20-33-02_resume_seed42_from_15000 \
  --checkpoint model_15999.pt \
  --num_envs 1



Full smoke-test command:

Quick Smoke-Test : 

python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Humanoid-v0 \
  --num_envs 4096 \
  --max_iterations 500 \
  --seed 42 \
  --run_name fresh_seed42 \
  --video \
  --video_length 300 \
  --headless \
  agent.save_interval=250

Resume example: 

python scripts/rsl_rl/train_eval.py \
  --task Velocity-Lilgreen-Humanoid-v0 \
  --num_envs 4096 \
  --max_iterations 501 \
  --seed 42 \
  --resume \
  --load_run 2026-07-01_20-00-41_fresh_seed42 \
  --checkpoint model_499.pt \
  --run_name resume_seed42_from_499 \
  --video \
  --video_length 300 \
  --headless \
  agent.save_interval=250






The practical rule:

Resume only:
  seed is optional

Resume for clean experiment continuity:
  use the same seed as the original run

Resume to explore a different continuation:
  use a different seed or omit it


Example normal playback:

python scripts/rsl_rl/play.py \
  --task Velocity-Lilgreen-Humanoid-v0 \
  --load_run 2026-07-01_12-45-00_fresh_seed42 \
  --checkpoint model_15000.pt \
  --num_envs 1

Example joystick playback:

python scripts/rsl_rl/play_joystick.py \
  --task Velocity-Lilgreen-Humanoid-v0 \
  --load_run 2026-07-01_12-45-00_fresh_seed42 \
  --checkpoint model_15000.pt \
  --num_envs 1

For export, the command stays separate:

python scripts/rsl_rl/export_policy.py \
  --task Velocity-Lilgreen-Humanoid-v0 \
  --load_run 2026-07-01_12-45-00_fresh_seed42 \
  --checkpoint model_15000.pt \
  --num_envs 1 \
  --headless