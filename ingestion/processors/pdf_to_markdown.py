import base64
import hashlib
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from core.utils.logger_utils import logger
from ingestion.processors import DEFAULT_IMAGES_OUTPUT_DIR


class PDFToMarkdownProcessor:
    def __init__(
            self,
            api_url: str,
            token: str,
            images_output_dir: str | Path = DEFAULT_IMAGES_OUTPUT_DIR,
    ):
        """
        Initialize the PDF to Markdown processor.

        Args:
            api_url (str): The API endpoint for PDF parsing.
            token (str): Authentication token for API access.
            images_output_dir (str | Path): Directory to save images extracted from markdown content.
        """
        self.api_url = api_url
        self.token = token
        self.images_output_dir = images_output_dir

        self.session = requests.Session()
        self._download_cache = {}
        self._ensure_dir(self.images_output_dir)

    @staticmethod
    def _ensure_dir(path: str | Path):
        """
        Create a directory if it does not already exist.

        Args:
            path (str | Path): The directory path to create.

        Raises:
            OSError: If the directory cannot be created.
        """
        try:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ready: {path}")
        except OSError as e:
            logger.error(f"Failed to create directory {path}: {e}")
            raise

    def parse_pdf(self, file_path: str | Path) -> str:
        """
        Parse a PDF file and convert it into a Markdown document.

        Workflow:
            1. Encode the PDF file into Base64 format.
            2. Send the encoded file to the parsing API.
            3. Process layout parsing results.
            4. Download and replace markdown images.
            5. Clean HTML (remove unnecessary div tags).

        Args:
            file_path (str | Path): Path to the input PDF file.

        Returns:
            str: The generated Markdown content.
        """
        encoded_file = self._encode_file(file_path)
        result = self._call_api(encoded_file)
        layout_results = result.get("layoutParsingResults", [])

        parts = []
        for res in layout_results:
            markdown_text = res.get("markdown", {}).get("text", "")
            images = res.get("markdown", {}).get("images", {})

            markdown_text = self._process_markdown_images(markdown_text, images)
            parts.append(markdown_text + "\n\n")

        full_markdown = self._clean_div("".join(parts))
        return full_markdown

    @staticmethod
    def _encode_file(file_path: str | Path) -> str:
        """
        Read a file and encode its contents into a Base64 string.

        Args:
            file_path (str | Path): Path to the file.

        Returns:
            str: Base64-encoded string of the file content.

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: If encoding fails.
        """
        try:
            return base64.b64encode(Path(file_path).read_bytes()).decode()
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to encode file {file_path}: {e}")
            raise

    def _call_api(self, file_data: str) -> dict:
        """
        Send a request to the PDF parsing API.

        Args:
            file_data (str): Base64-encoded file content.

        Returns:
            dict: Parsed result returned by the API.

        Raises:
            HTTPError: If the request fails.
            ValueError: If expected fields are missing in the API response.
        """
        headers = {
            "Authorization": f"token {self.token}",
            "Content-Type": "application/json",
        }
        required_payload = {
            "file": file_data,
            "fileType": 0,
        }

        optional_payload = {
            "markdownIgnoreLabels": [
                "header",
                "header_image",
                "footer",
                "footer_image",
                "number",
                "footnote",
                "aside_text"
            ],
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useLayoutDetection": True,
            "useChartRecognition": False,
            "useSealRecognition": True,
            "useOcrForImageBlock": False,
            "mergeTables": True,
            "relevelTitles": True,
            "layoutShapeMode": "auto",
            "promptLabel": "ocr",
            "repetitionPenalty": 1,
            "temperature": 0,
            "topP": 1,
            "minPixels": 147384,
            "maxPixels": 2822400,
            "layoutNms": True,
            "restructurePages": True
        }

        payload = {**required_payload, **optional_payload}

        response = self.session.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=300,
        )
        response.raise_for_status()

        data = response.json()
        if "result" not in data:
            logger.error(f"Invalid API response: {data}")
            raise ValueError("Missing 'result' in API response")

        return data["result"]

    def _process_markdown_images(self, markdown_text: str, images: dict) -> str:
        """
        Download images referenced in markdown and replace their paths.

        Args:
            markdown_text (str): Original markdown content.
            images (dict): Mapping of original image paths to URLs.

        Returns:
            str: Updated markdown content with local image paths.
        """
        for old_path, url in images.items():
            logger.info(f"Download image {url}")
            new_path = self._download_image(url, self.images_output_dir)
            markdown_text = markdown_text.replace(old_path, new_path)
        return markdown_text

    def _download_image(self, url: str, save_dir: str | Path) -> str:
        """
        Download an image from a URL and save it locally.

        Args:
            url (str): Image URL.
            save_dir (str | Path): Directory to save the image.

        Returns:
            str: Local file path of the saved image.
        """
        if url in self._download_cache:
            return self._download_cache[url]

        response = self.session.get(url, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        ext = self._get_extension(url, content_type)

        image_bytes = response.content
        basename = hashlib.md5(image_bytes).hexdigest()
        save_path = Path(save_dir) / f"{basename}{ext}"

        if save_path.exists():
            result = str(save_path)
            self._download_cache[url] = result
            return result

        tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
        try:
            tmp_path.write_bytes(image_bytes)
            tmp_path.replace(save_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        result = str(save_path)
        self._download_cache[url] = result
        return result

    @staticmethod
    def _get_extension(url: str, content_type: str) -> str:
        """
        Determine the file extension of a resource based on its URL and HTTP Content-Type.

        This method uses a fallback strategy to infer the most appropriate file extension:

            1. Try to extract the extension directly from the URL path.
            2. If not available, infer the extension from the Content-Type header.
            3. If both methods fail, default to ".jpg".

        Args:
            url (str): The resource URL. May or may not contain a valid file extension.
            content_type (str): The HTTP Content-Type header value.

        Returns:
            str: File extension including the leading dot (e.g., ".jpg", ".png").
        """
        path = urlparse(url).path
        ext = Path(path).suffix.lower()

        if ext and ext.startswith(".") and 1 < len(ext) <= 5:
            return ext

        ext = mimetypes.guess_extension(content_type.split(";")[0])
        if ext:
            return ext

        return ".jpg"

    @staticmethod
    def _clean_div(markdown_text: str) -> str:
        """
        Remove all <div> tags while preserving their inner content.

        Behavior:
            - Removes all <div> tags.
            - Preserves all inner content, including text, <img>, <table>, etc.
            - Does not alter document structure beyond removing <div> wrappers.

        Args:
            markdown_text (str): Markdown content containing HTML.

        Returns:
            str: Cleaned markdown content.
        """
        soup = BeautifulSoup(markdown_text, "html.parser")

        for div in soup.find_all("div")[::-1]:
            div.unwrap()

        return str(soup)
