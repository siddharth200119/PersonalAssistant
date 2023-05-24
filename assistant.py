import gpt4all

gptj = gpt4all.GPT4All("ggml-mpt-7b-chat")

message = input("enter message: ")
messages = [{"role": "user", "content": message}]


gptj.generate(message)