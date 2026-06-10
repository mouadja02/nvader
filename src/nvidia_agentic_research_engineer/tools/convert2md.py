from markitdown import MarkItDown
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def pdf_to_markdown(path: str) -> str:    
    md = MarkItDown(
        enable_plugins=True,
        llm_client=OpenAI(api_key=os.environ.get("NVIDIA_API_KEY", ""), base_url="https://integrate.api.nvidia.com/v1"),
        llm_model="nvidia/nemotron-nano-12b-v2-vl",
    )
    result = md.convert_local(path)
    return result.text_content