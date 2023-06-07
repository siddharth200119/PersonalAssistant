from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

PATH = "model/ggml-gpt4all-l13b-snoozy.bin"

llm = GPT4All(model=PATH, verbose=True)

template = PromptTemplate(input_variables=['question'], template="""
    ### Instruction: 
    This prompt below is a question to a answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.
    
    ### Prompt: 
    {question}
    
    ### Response:""")

llm_chain = LLMChain(prompt=template, llm=llm, verbose = True)
conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=ConversationBufferMemory()
)

while True:
    prompt = input("enter: ")
    print(conversation.predict(input = prompt))
