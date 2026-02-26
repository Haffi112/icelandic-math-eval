"""
Image Handler for Math Problems
Loads and encodes images for API calls.
"""

import base64
from pathlib import Path
from typing import Optional
from PIL import Image
import io


class ImageHandler:
    """Handles image loading and encoding for vision-enabled models."""

    @staticmethod
    def encode_image_to_base64(image_path: str) -> Optional[str]:
        """
        Load image and encode it to base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded string of the image, or None if failed
        """
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode("utf-8")
                return base64_image
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    @staticmethod
    def get_image_mime_type(image_path: str) -> str:
        """
        Get MIME type from image file extension.

        Args:
            image_path: Path to the image file

        Returns:
            MIME type string (e.g., 'image/png')
        """
        ext = Path(image_path).suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(ext, "image/png")

    @staticmethod
    def create_image_content(image_path: str) -> Optional[dict]:
        """
        Create OpenAI-compatible image content dictionary.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with image data for API call, or None if failed
        """
        base64_image = ImageHandler.encode_image_to_base64(image_path)
        if not base64_image:
            return None

        mime_type = ImageHandler.get_image_mime_type(image_path)

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            }
        }

    @staticmethod
    def validate_image(image_path: str) -> bool:
        """
        Validate that image exists and can be opened.

        Args:
            image_path: Path to the image file

        Returns:
            True if valid, False otherwise
        """
        try:
            path = Path(image_path)
            if not path.exists():
                print(f"Image file not found: {image_path}")
                return False

            # Try to open with PIL to verify it's a valid image
            with Image.open(image_path) as img:
                img.verify()

            return True
        except Exception as e:
            print(f"Invalid image {image_path}: {e}")
            return False
