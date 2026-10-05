# QwenLean

**Cut the waste. Keep the reasoning.**

Experimental research on Qwen3.5-0.8B's accuracy versus reasoning compute. We measure
literal loops, redundant reasoning and poor termination separately. Token count alone
is not a sufficient objective: correctness comes first and difficult problems may
need long traces.

Current status: environment validation and real-inference baseline under construction.
No training has run and no improvement has been established.

Python 3.12, uv, Apple Silicon. No Docker. Generated weights and run outputs stay out of Git.
