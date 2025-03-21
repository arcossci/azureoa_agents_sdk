from openai import AsyncAzureOpenAI
from agents import set_default_openai_client
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create OpenAI client using Azure OpenAI
openai_client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)

# Set the default OpenAI client for the Agents SDK
set_default_openai_client(openai_client)

from agents import Agent
from openai.types.chat import ChatCompletionMessageParam

# Create a banking assistant agent
banking_assistant = Agent(
    name="Banking Assistant",
    instructions="You are a helpful banking assistant. Be concise and professional.",
    model="gpt-4o",  # This will use the deployment specified in your Azure OpenAI/APIM client
    tools=[check_account_balance]  # A function tool defined elsewhere
)


