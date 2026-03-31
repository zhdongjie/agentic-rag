from langchain_core.prompts import ChatPromptTemplate


def _build_chat_prompt(system_lines: list[str], human_content: str | list[dict]) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", "\n".join(system_lines)),
        ("human", human_content),
    ])


CODE_EXPLANATION_PROMPT = _build_chat_prompt(
    system_lines=[
        "你是一个知识库预处理助手。",
        "请为目标代码块生成简洁、自然的中文说明。",
        "滑动窗口上下文只用于帮助你理解目标代码块。",
        "重点说明这段代码做了什么、关键逻辑是什么、在当前上下文中起什么作用。",
        "不要评价它对检索是否有价值，也不要评价它是否重要、是否有用。",
        "避免空洞的总结句，例如“因此很适合检索”之类的话。",
        "不要照抄原始代码，只输出最终说明文本。",
    ],
    human_content="\n".join([
        "目标类型：代码",
        "chunk_index: {chunk_index}",
        "heading_hierarchy: {heading_hierarchy}",
        "",
        "目标代码：",
        "{code}",
        "",
        "滑动窗口上下文：",
        "{window_context}",
        "",
        "请说明这段代码做了什么、关键逻辑是什么，以及它在当前上下文中解决了什么问题。",
    ]),
)

TABLE_EXPLANATION_PROMPT = _build_chat_prompt(
    system_lines=[
        "你是一个知识库预处理助手。",
        "请为目标 Markdown 表格生成简洁、自然的中文说明。",
        "滑动窗口上下文只用于帮助你理解目标表格。",
        "重点说明表格主题、重要维度，以及它在当前上下文中的核心结论。",
        "不要评价它对检索是否有价值，也不要评价它是否重要、是否有用。",
        "避免空洞的总结句，例如“因此很适合检索”之类的话。",
        "不要重复原始表格内容，只输出最终说明文本。",
    ],
    human_content="\n".join([
        "目标类型：表格",
        "chunk_index: {chunk_index}",
        "heading_hierarchy: {heading_hierarchy}",
        "",
        "目标表格：",
        "{table}",
        "",
        "滑动窗口上下文：",
        "{window_context}",
        "",
        "请说明表格的主题、各列或各维度分别表示什么，以及它的核心结论是什么。",
    ]),
)

IMAGE_EXPLANATION_PROMPT = _build_chat_prompt(
    system_lines=[
        "你是一个知识库预处理助手。",
        "请为目标图片块生成简洁、自然的中文说明。",
        "你可以同时看到图片和滑动窗口上下文。",
        "请结合两者说明图片传达了什么，以及它和当前文档内容的关系。",
        "不要评价它对检索是否有价值，也不要评价它是否权威、是否重要、是否有用。",
        "避免空洞的总结句，例如“因此很适合检索”或类似表述。",
        "如果图片是代码截图、源码截图、表格截图或界面截图，请说明它支撑了哪个具体技术点，不要空泛夸赞。",
        "只输出最终说明文本。",
    ],
    human_content=[
        {
            "type": "text",
            "text": "\n".join([
                "目标类型：图片",
                "chunk_index: {chunk_index}",
                "heading_hierarchy: {heading_hierarchy}",
                "",
                "滑动窗口上下文：",
                "{window_context}",
                "",
                "请说明这张图片传达了什么，以及它和附近上下文之间的关系。",
            ]),
        },
        {
            "type": "image_url",
            "image_url": {"url": "{image_data_url}"},
        },
    ],
)
