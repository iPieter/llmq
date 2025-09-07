import yaml
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from pathlib import Path


class PipelineStage(BaseModel):
    """Configuration for a single pipeline stage."""

    name: str = Field(description="Stage name (must be unique within pipeline)")
    worker: str = Field(
        description="Worker type (e.g., 'vllm', 'dummy', 'bloom-filter')"
    )
    config: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Stage-specific configuration"
    )

    @validator("name")
    def name_must_be_valid(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Stage name must be a non-empty string")
        # Ensure name is safe for queue names (no spaces, special chars)
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                "Stage name can only contain letters, numbers, hyphens, and underscores"
            )
        return v


class PipelineConfig(BaseModel):
    """Pipeline configuration definition."""

    name: str = Field(description="Pipeline name")
    stages: List[PipelineStage] = Field(
        min_length=1, description="Ordered list of pipeline stages"
    )
    config: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Global pipeline configuration"
    )

    @validator("name")
    def name_must_be_valid(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Pipeline name must be a non-empty string")
        # Ensure name is safe for queue names
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                "Pipeline name can only contain letters, numbers, hyphens, and underscores"
            )
        return v

    @validator("stages")
    def stages_must_have_unique_names(cls, v):
        names = [stage.name for stage in v]
        if len(names) != len(set(names)):
            raise ValueError("All stage names must be unique within a pipeline")
        return v

    @classmethod
    def from_yaml_file(cls, path: Path) -> "PipelineConfig":
        """Load pipeline configuration from YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Pipeline configuration file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("Pipeline configuration must be a YAML object")

        return cls(**data)

    @classmethod
    def from_yaml_string(cls, yaml_str: str) -> "PipelineConfig":
        """Load pipeline configuration from YAML string."""
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("Pipeline configuration must be a YAML object")

        return cls(**data)

    def get_stage_queue_name(self, stage_name: str) -> str:
        """Get the queue name for a specific stage."""
        return f"pipeline.{self.name}.{stage_name}"

    def get_pipeline_exchange_name(self) -> str:
        """Get the pipeline routing exchange name."""
        return f"pipeline.{self.name}.routing"

    def get_stage_routing_key(self, stage_name: str) -> str:
        """Get the routing key for stage completion."""
        return f"pipeline.{self.name}.stage.{stage_name}.complete"

    def get_stage_by_name(self, stage_name: str) -> Optional["PipelineStage"]:
        """Get a stage by its name."""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None

    def get_pipeline_results_queue_name(self) -> str:
        """Get the final results queue name for the pipeline."""
        return f"pipeline.{self.name}.results"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.model_dump()


def load_pipeline_config(path: Path) -> PipelineConfig:
    """Convenience function to load pipeline configuration."""
    return PipelineConfig.from_yaml_file(path)


# Example pipeline configurations for testing
EXAMPLE_DOCUMENT_PIPELINE = """
name: document-processing
stages:
  - name: deduplication
    worker: bloom-filter
    config:
      bloom_size: 1000000
      error_rate: 0.001
  - name: translation
    worker: vllm
    config:
      model: microsoft/DialoGPT-medium
  - name: formatting
    worker: vllm
    config:
      model: microsoft/DialoGPT-medium
      temperature: 0.1
config:
  timeout_minutes: 60
  retry_count: 3
"""

EXAMPLE_SIMPLE_PIPELINE = """
name: simple-test
stages:
  - name: stage1
    worker: dummy
  - name: stage2
    worker: dummy
"""
