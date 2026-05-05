from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()#to load env variable from .env file

class BedrockLLM:
    MODELS = {
        "claude": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-opus": "anthropic.claude-3-opus-20240229-v1:0",
    "llama": "meta.llama3-70b-instruct-v1:0",
    "mistral": "mistral.mistral-large-2402-v1:0",
    }
    
    def __init__(self, model: str = "claude", region: str = "us-east-1"):
        if model not in self.MODELS:
            raise ValueError(f"Unknown model. Choose from: {list(self.MODELS.keys())}")
        
        self.llm = ChatBedrock(
            model_id=self.MODELS[model],
            region_name=region,
            model_kwargs={"temperature": 0.5, "max_tokens": 1000},
        )
    
    def ask(self, question: str, system_prompt: str = "You are a helpful assistant.") -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question})


def get_llm_answer(model: str, question: str) -> str:
    return BedrockLLM(model=model).ask(question)


# Usage
# print(get_llm_answer("claude", "What is a LangChain Runnable?"))