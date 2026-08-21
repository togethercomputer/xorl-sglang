# Single source of truth for the sgl-eval commit every CI variant installs.
# Meant to be sourced, not executed -- each variant then installs
# "$SGL_EVAL_SPEC" with its own pip invocation, since those differ (uv pip on
# CUDA/CPU, `docker exec ... pip` on AMD, `python3 -m pip` on NPU).
#
# sgl-eval is git-only and cannot be declared in python/pyproject.toml (see the
# note there). Every eval that shells out to the `sgl-eval` CLI fails without
# it, and a bump moves scoring for all of them at once -- so re-baseline
# MODEL_SCORE_THRESHOLDS in
# test/registered/eval/test_text_models_gsm8k_eval.py, and the mmlu thresholds
# of run_eval's other callers, before changing this.
#
# Backported to this fork's v0.5.17 base: upstream added this file after the
# release branch was cut, so _pr-test-stage-cpu.yml sources a script that does
# not exist here and the CPU lane dies with
#   scripts/ci/utils/sgl_eval_ref.sh: No such file or directory
#
# The ref below is v0.5.17's, NOT the one in upstream main's copy of this file
# (6690895609dcbc5df1e7b00dd57c9502b868ec4d). Taking main's would put the CPU
# lane on a different sgl-eval than ci_install_dependency.sh installs on the
# CUDA lanes, against thresholds calibrated for this one.
SGL_EVAL_REF="b2a2703c42cae379bbcb8b7ff092df6601a61694"
SGL_EVAL_SPEC="sgl-eval@git+https://github.com/sgl-project/sgl-eval.git@${SGL_EVAL_REF}"
