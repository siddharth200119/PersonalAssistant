import streamlit as st 
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All

PATH = "model/ggml-gpt4all-l13b-snoozy.bin"

llm = GPT4All(model=PATH, verbose=True)

template = PromptTemplate(input_variables=['question'], template="""
    ### Instruction: 
    This prompt below is a question to a answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.
    
    ### Prompt: 
    {question}
    
    ### Response:""")

llm_chain = LLMChain(prompt=template, llm=llm, verbose = True)

st.title('🦜🔗 GPT4ALL Y\'All')
st.info('This is using the MPT model!')
prompt = st.text_input('Enter your prompt here!')

if prompt: 
    response = llm_chain.run(prompt)
    print(response)
    st.write(response)
