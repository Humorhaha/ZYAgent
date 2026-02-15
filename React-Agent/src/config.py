from dataclasses import dataclass
import os
from dotenv import load_dotenv


@dataclass
class Settings:
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    max_steps: int
    temperature: float



def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "8")),
        temperature=float(os.getenv("AGENT_TEMPERATURE", "0.1")),
    )
