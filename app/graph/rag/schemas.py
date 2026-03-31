from pydantic import BaseModel, Field


class QueryTransformResult(BaseModel):
    rewritten_query: str = Field(description="更完整、更适合检索的主查询。")
    query_type: str = Field(description="问题类型，例如 specific、overview、comparison、procedure、reason。")
    sub_queries: list[str] = Field(
        default_factory=list,
        description="补充召回用的子查询；如果主查询已足够，应返回空数组。",
    )
    retrieval_focus: list[str] = Field(
        default_factory=list,
        description="检索重点，例如 definition、mechanism、comparison、example、impact。",
    )


class RetrievalRelevanceEvaluation(BaseModel):
    passed: bool = Field(description="检索结果是否足以支撑后续回答。")
    score: int = Field(description="检索结果对回答问题的支撑程度，范围为 0 到 100。")
    reason: str = Field(description="对判断结果的简短说明。")
    missing_aspects: list[str] = Field(
        default_factory=list,
        description="检索结果缺失的关键点；如果没有缺失，返回空数组。",
    )


class AnswerEvaluation(BaseModel):
    passed: bool = Field(description="当前答案是否可以直接返回给用户。")
    score: int = Field(description="答案质量评分，范围为 0 到 100。")
    reason: str = Field(description="对评估结果的简短说明。")
    issues: list[str] = Field(
        default_factory=list,
        description="答案存在的主要问题；如果没有问题，返回空数组。",
    )
