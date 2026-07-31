import base64
import re
from pathlib import Path

MARKDOWN_FILE = Path("Master Notebook - Null Pointers.md")
OUTPUT_FILE = Path("Master Notebook - Null Pointers-clean.md")
IMAGE_DIR = Path("images")


def save_image(encoded: str, extension: str, index: int) -> str:
    encoded = re.sub(r"\s+", "", encoded)

    extension = {
        "jpeg": "jpg",
        "svg+xml": "svg",
    }.get(extension, extension)

    filename = f"markdown_image_{index:03d}.{extension}"
    destination = IMAGE_DIR / filename

    if extension == "svg":
        decoded = base64.b64decode(encoded).decode("utf-8")
        destination.write_text(decoded, encoding="utf-8")
    else:
        destination.write_bytes(base64.b64decode(encoded))

    print(f"Created: {destination}")
    return destination.as_posix()


def main() -> None:
    if not MARKDOWN_FILE.exists():
        raise FileNotFoundError(f"找不到文件：{MARKDOWN_FILE}")

    IMAGE_DIR.mkdir(exist_ok=True)

    text = MARKDOWN_FILE.read_text(encoding="utf-8")
    image_index = 0

    # 处理 Markdown 格式：
    # ![description](data:image/png;base64,...)
    markdown_pattern = re.compile(
        r"!\[([^\]]*)\]\("
        r"data:image/(png|jpeg|jpg|gif|webp|svg\+xml);base64,"
        r"([A-Za-z0-9+/=\r\n]+)"
        r"\)",
        re.IGNORECASE,
    )

    def replace_markdown(match: re.Match) -> str:
        nonlocal image_index

        alt_text = match.group(1) or f"Image {image_index + 1}"
        mime_extension = match.group(2).lower()
        encoded = match.group(3)

        path = save_image(encoded, mime_extension, image_index)
        image_index += 1

        return f"![{alt_text}]({path})"

    text, markdown_count = markdown_pattern.subn(replace_markdown, text)

    # 处理 HTML 格式：
    # <img src="data:image/png;base64,..." ...>
    html_pattern = re.compile(
        r"<img\b([^>]*?)src=[\"']"
        r"data:image/(png|jpeg|jpg|gif|webp|svg\+xml);base64,"
        r"([A-Za-z0-9+/=\r\n]+)"
        r"[\"']([^>]*)>",
        re.IGNORECASE,
    )

    def replace_html(match: re.Match) -> str:
        nonlocal image_index

        before_src = match.group(1)
        mime_extension = match.group(2).lower()
        encoded = match.group(3)
        after_src = match.group(4)

        path = save_image(encoded, mime_extension, image_index)
        image_index += 1

        attributes = f"{before_src}{after_src}".strip()
        space = " " if attributes else ""

        return f'<img src="{path}"{space}{attributes}>'

    text, html_count = html_pattern.subn(replace_html, text)

    OUTPUT_FILE.write_text(text, encoding="utf-8")

    print()
    print(f"Markdown 图片替换数量：{markdown_count}")
    print(f"HTML 图片替换数量：{html_count}")
    print(f"总共提取图片：{image_index}")
    print(f"新文件：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()