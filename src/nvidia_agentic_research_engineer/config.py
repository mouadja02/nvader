from pathlib import Path
from pydantic import BaseModel, Field
import toml

PROJECT_TOML_PATH = Path("pyproject.toml")
toml_content = toml.load(PROJECT_TOML_PATH) if PROJECT_TOML_PATH.exists() else {}

# Load project configuration from pyproject.toml
class ProjectTOML(BaseModel):
    name: str = toml_content["project"]["name"] if "name" in toml_content.get("project", {}) else "NVIDIA Agentic Research Engineer" 
    description: str = toml_content["project"]["description"] if "description" in toml_content.get("project", {}) else ""
    version: str = toml_content["project"]["version"] if "version" in toml_content.get("project", {}) else ""
    authors_names: list = [author.get("name", "") for author in toml_content["project"]["authors"]] if "authors" in toml_content.get("project", {}) else []

class AppConfig(BaseModel):
    project_name : str = "NVIDIA Agentic Research Engineer"
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    vector_store_dir: Path = Path("data/vector_store")
    traces_dir: Path = Path("data/traces")
    default_top_k: int = 5
    require_citations: bool = True
    max_retries: int = 3