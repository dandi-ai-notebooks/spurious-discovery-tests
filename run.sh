#!/bin/bash

./run_test.py alpha_attention_01 prompt-1 google/gemini-2.5-flash-preview
./run_test.py alpha_attention_01 prompt-1 google/gemini-2.5-pro-preview
./run_test.py alpha_attention_01 prompt-1 openai/gpt-4.1
./run_test.py alpha_attention_01 prompt-1 openai/chatgpt-4o-latest
./run_test.py alpha_attention_01 prompt-1 anthropic/claude-3.5-sonnet
./run_test.py alpha_attention_01 prompt-1 anthropic/claude-3.7-sonnet
./run_test.py alpha_attention_01 prompt-1 deepseek/deepseek-chat-v3-0324

./run_test.py alpha_attention_01 prompt-2 google/gemini-2.5-flash-preview
./run_test.py alpha_attention_01 prompt-2 google/gemini-2.5-pro-preview
./run_test.py alpha_attention_01 prompt-2 openai/gpt-4.1
./run_test.py alpha_attention_01 prompt-2 openai/chatgpt-4o-latest
./run_test.py alpha_attention_01 prompt-2 anthropic/claude-3.5-sonnet
./run_test.py alpha_attention_01 prompt-2 anthropic/claude-3.7-sonnet
./run_test.py alpha_attention_01 prompt-2 deepseek/deepseek-chat-v3-0324

./run_test.py theta_memory_load_01 prompt-1 google/gemini-2.5-flash-preview
./run_test.py theta_memory_load_01 prompt-1 google/gemini-2.5-pro-preview
./run_test.py theta_memory_load_01 prompt-1 openai/gpt-4.1
./run_test.py theta_memory_load_01 prompt-1 openai/chatgpt-4o-latest
./run_test.py theta_memory_load_01 prompt-1 anthropic/claude-3.5-sonnet
./run_test.py theta_memory_load_01 prompt-1 anthropic/claude-3.7-sonnet
./run_test.py theta_memory_load_01 prompt-1 deepseek/deepseek-chat-v3-0324
