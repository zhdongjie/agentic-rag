import hashlib
from pathlib import Path

from core.utils.logger_utils import logger


def list_files_with_allowed_extensions(
        data_path: str | Path,
        allowed_types: tuple[str, ...],
        recursive: bool = False
) -> tuple[str, ...]:
    """
    List all files in a directory that match the given file extensions.

    This function scans a specified directory and returns all file paths
    whose suffix (file extension) is included in the allowed_types list.
    It supports both recursive and non-recursive traversal.

    Args:
        data_path (str | Path): Path to the directory to search.
        allowed_types (tuple[str, ...]): Tuple of allowed file extensions
            (e.g., ('.jpg', '.png', '.pdf')). Matching is case-insensitive.
        recursive (bool): Whether to recursively search subdirectories.
            - True: Search all nested directories.
            - False: Only search the top-level directory.

    Returns:
        tuple[str, ...]: A tuple of file paths (as strings) that match
        the allowed extensions. Returns an empty tuple if:
            - The directory does not exist
            - No matching files are found
    """
    path = Path(data_path)
    if not path.is_dir():
        logger.error(f"[list_files_with_allowed_extensions] '{data_path}' is not a directory")
        return ()

    allowed_types = tuple(ext.lower() for ext in allowed_types)

    if recursive:
        files = (
            str(p)
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in allowed_types
        )
    else:
        files = (
            str(p)
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in allowed_types
        )

    return tuple(files)


def get_file_content_md5_hex(file_path: str | Path) -> str | None:
    """
    Compute the MD5 hash (hex digest) of a file's content.

    This function reads the file in chunks to avoid loading large files
    entirely into memory, making it suitable for large file processing.

    Args:
        file_path (str | Path): Path to the target file.

    Returns:
        str | None:
            - MD5 hex string (32-character lowercase string) if successful
            - None if:
                - File does not exist
                - Path is not a file
                - An error occurs during reading

    Raises:
        None directly. All exceptions are caught and logged internally.

    Example:
        >>> get_file_content_md5_hex("example.txt")
        '5d41402abc4b2a76b9719d911017c592'
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"[get_file_content_md5_hex] file '{file_path}' not exist")
        return None

    if not path.is_file():
        logger.error(f"[get_file_content_md5_hex] '{file_path}' is not a file")
        return None

    md5_ins = hashlib.md5()
    chunk_size = 4096

    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                md5_ins.update(chunk)
            md5_hex = md5_ins.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"[get_file_md5_hex] get file md5 hex failed: {str(e)}")
        return None
