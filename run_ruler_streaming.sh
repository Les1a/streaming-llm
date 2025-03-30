bash scripts/run_streaming.sh longchat-7b-32k-streaming synthetic
bash scripts/run_streaming.sh longchat-7b-32k synthetic
bash scripts/run_streaming.sh longchat-7b-32k-ours synthetic

# Adjust streaming-llm hyperparameters in scripts/run_streaming.sh:
# 
# export RECENT_SIZE_RATIO=?
# export START_SIZE=?


# Adjust test task length in scripts/config_models.sh:
# 
# SEQ_LENGTHS=(
#     # 65536
#     # 57344
#     # 49152
#     # 40960
#     # 32768
#     # 24576
#     # 16384
#     # 8192
#     4096
# )

# Adjust test task sample numbers in scripts/config_tasks.sh:
# 
# NUM_SAMPLES=?


# Adjust streaming-llm model architecture in scripts/pred/model_wrappers.py:
# 
# class StreamingLongChatModel: