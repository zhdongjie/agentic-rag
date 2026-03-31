from pathlib import Path
from typing import Iterable


def get_root_path(markers: Iterable[str] = ("pyproject.toml", ".git")) -> Path:
    """
    Determine the project root directory by searching for marker files.

    This function traverses upward from the current file location and attempts
    to identify the project root directory by checking for the existence of one
    or more predefined "marker" files or directories (e.g., "pyproject.toml",
    ".git"). The first parent directory containing any of the specified markers
    is considered the project root.

    Args:
        markers (Iterable[str]): A collection of file or directory names used
            as indicators of the project root. Common examples include:
                - "pyproject.toml" (modern Python projects)
                - ".git" (Git repository root)

    Returns:
        Path: The resolved project root directory as a pathlib.Path object.

    Example:
        >>> root = get_root_path()
        >>> print(root)
        PosixPath('/path/to/project')

        >>> root = get_root_path(markers=(".git",))
        >>> print(root)
        PosixPath('/path/to/git/repo')
    """
    path = Path(__file__).resolve()

    for parent in path.parents:
        if any((parent / marker).exists() for marker in markers):
            return parent

    return path.parent


def get_abs_path(relative_path: str) -> Path:
    """
    Convert a project-relative path into an absolute path.

    This function resolves a given relative path based on the detected
    project root directory.

    Args:
        relative_path (str): Path relative to the project root.

    Returns:
        str: Absolute path corresponding to the given relative path.

    Example:
        >>> get_abs_path("data/file.txt")
        '/absolute/path/to/project/data/file.txt'
    """
    root_path = get_root_path()
    return root_path / relative_path


def to_absolute_path(main_file_path: str | Path, relative_path: str) -> Path:
    """
    Convert a relative path into an absolute path based on a given file.

    This function resolves a relative path using the directory of a specified
    "main" file (typically __file__).

    Args:
        main_file_path (str | Path): Path to the reference file (e.g., __file__).
        relative_path (str): Relative path to resolve.

    Returns:
        str: Absolute path resolved from the reference file's directory.

    Example:
        >>> to_absolute_path(__file__, "../data/file.txt")
        '/absolute/path/to/data/file.txt'
    """
    main_path = Path(main_file_path)
    abs_path = (main_path.parent / relative_path).resolve()
    return abs_path
