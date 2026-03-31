import json
import re


class JavaDocProcessor:
    def __init__(self, content: str):
        self.content = content

    def process(self) -> tuple[str, dict]:
        extracted_meta = {}
        START_TAG = "<" + "!-- QUESTIONS_START --" + ">"
        END_TAG = "<" + "!-- QUESTIONS_END --" + ">"

        if START_TAG not in self.content:
            extracted_meta["questions"] = "[]"
            return self.content, extracted_meta

        content_main, rest = self.content.split(START_TAG, 1)
        q_part, content_tail = rest.split(END_TAG, 1) if END_TAG in rest else (rest, "")

        questions = []
        rag_question_text = ""

        blocks = re.split(r'###\s+', q_part)

        for b in blocks:
            if not b.strip():
                continue

            if "URL:" not in b:
                rag_question_text += f"\n### {b}"
                continue

            lines = [l.strip() for l in b.split("\n") if l.strip()]
            title = lines[0]
            url = next((l.replace("URL:", "").strip() for l in lines if l.startswith("URL:")), "")

            if url:
                questions.append({"title": title, "url": url})

            clean_block = re.sub(r"URL:\s*(https?://[^\s]+)", "", b).strip()
            rag_question_text += f"\n### {clean_block}\n"

        extracted_meta["questions"] = json.dumps(questions, ensure_ascii=False)

        final_content = (
                content_main
                + START_TAG
                + rag_question_text
                + "\n"
                + END_TAG
                + content_tail
        )

        return final_content.strip(), extracted_meta