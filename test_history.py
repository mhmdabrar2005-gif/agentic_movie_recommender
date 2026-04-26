from agents import run_agent

history = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "Hello. How are you today?"},
    {"role": "user", "content": "feeling good"},
    {"role": "assistant", "content": "That's great to hear. What kind of movies or TV shows are you in the mood for today?"}
]

import traceback

try:
    print(run_agent("something romantic classical", history=history))
except Exception as e:
    traceback.print_exc()
