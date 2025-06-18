from typing import Optional, Dict

from pydantic import BaseModel, Field, ConfigDict


class GenerationOptions(BaseModel):
    """LLM生成的相关选项"""

    model_config = ConfigDict(extra="allow")

    temperature: Optional[float] = Field(
        title="采样温度",
        description="如果为空或者0，表示不采样",
        default=None,
        ge=0.0
    )
    max_tokens: int = Field(
        title="最大Token数",
        description="该次生成过程中所使用的最大Token数量",
        default=4096
    )
    stream: bool = False
    presence_penalty: Optional[float] = None
    top_p: Optional[int] = None
    extra_body: Optional[Dict] = None


class LLMConfig(BaseModel):
    """大模型服务配置"""

    base_url: str = Field(
        title="服务地址",
        description="服务地址"
    )
    model: str = Field(
        title="模型名称",
        description="一个地址可能对应多个模型，需要指定模型名称"
    )
    api_key: Optional[str] = Field(
        title="API密钥",
        description="如果启动大模型服务的时候没有指定，这里就不需要",
        default=None
    )