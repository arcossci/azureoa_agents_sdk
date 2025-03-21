from openai import AsyncAzureOpenAI
from agents import set_default_openai_client, Agent, Runner, WebSearchTool
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables
load_dotenv()

# Create OpenAI client using Azure OpenAI
openai_client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
)

# Set the default OpenAI client for the Agents SDK
set_default_openai_client(openai_client)

# Define research agents with WebSearchTool
agente_investigacion = Agent(
    model="gpt-4o-mini",
    name="Agente de Investigación Legal",
    instructions="Realiza investigaciones utilizando la herramienta WebSearchTool sobre el tema de tecnología legal en la construcción en Colombia",
    tools=[WebSearchTool()]
)

# Define question answering agent
agente_respuestas = Agent(
    model="gpt-4o-mini",
    name="Agente de Respuestas Legales",
    instructions="Responde preguntas planteadas con base en los hallazgos del agente de investigación sobre tecnología legal en la construcción en Colombia",
)

# Define triage agent to handoff tasks to research agents and question answering agent
agente_triage = Agent(
    model="gpt-4o-mini",
    name="Agente de Triaje",
    instructions="Primero, el agente de investigación realizará sus búsquedas. Luego, el agente de respuestas responderá preguntas basadas en los hallazgos.",
    handoffs=[agente_investigacion, agente_respuestas],
)

async def main():
    while True:
        user_input = input("Ingrese su mensaje (escriba 'salir' para terminar): ")
        if user_input.lower() == "salir":
            break
        result = await Runner.run(agente_triage, input=user_input)
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())