"""
EXIF / metadata analysis utilities.

Extracts metadata from image files and checks for signs of digital
manipulation (missing timestamps, editing software signatures, etc.).
"""

from __future__ import annotations

from PIL import Image
from PIL.ExifTags import TAGS


# Software names commonly associated with image editing / generation
_EDITING_SOFTWARE_KEYWORDS = [
    "photoshop",
    "gimp",
    "affinity",
    "lightroom",
    "canva",
    "stable diffusion",
    "midjourney",
    "dall-e",
    "dall·e",
    "firefly",
    "pixelmator",
    "capture one",
    "luminar",
    "topaz",
]


class MetadataAnalyzer:
    """
    Analyzes EXIF metadata of an image for manipulation indicators.

    Args:
        image_path: Path to the image file to analyse.
    """

    def __init__(self, image_path: str):
        self.image_path = image_path
        self._exif_raw: dict = {}
        self._exif_human: dict = {}
        self._loaded = False

    # ── private helpers ──────────────────────

    def _load(self) -> None:
        """Load EXIF data from the image (lazy)."""
        if self._loaded:
            return
        try:
            with Image.open(self.image_path) as img:
                raw = img._getexif()  # returns None for non-JPEG or no EXIF
        except Exception:
            raw = None

        if raw:
            self._exif_raw = raw
            self._exif_human = {
                TAGS.get(tag, str(tag)): value for tag, value in raw.items()
            }
        self._loaded = True

    # ── public API ───────────────────────────

    @property
    def exif(self) -> dict:
        """Return the human-readable EXIF dictionary."""
        self._load()
        return dict(self._exif_human)

    def has_exif(self) -> bool:
        """Return True if the image contains any EXIF metadata."""
        self._load()
        return bool(self._exif_human)

    def get_creation_date(self) -> str | None:
        """
        Return the original creation date stored in EXIF, or None if absent.
        """
        self._load()
        return self._exif_human.get("DateTimeOriginal") or self._exif_human.get(
            "DateTime"
        )

    def get_camera_make_model(self) -> dict:
        """Return the camera make and model from EXIF, if available."""
        self._load()
        return {
            "make": self._exif_human.get("Make"),
            "model": self._exif_human.get("Model"),
        }

    def get_software(self) -> str | None:
        """Return the software tag from EXIF, if available."""
        self._load()
        return self._exif_human.get("Software")

    def detect_editing_software(self) -> dict:
        """
        Check whether the EXIF *Software* tag suggests image editing.

        Returns:
            dict with keys:
            - ``detected``  : bool — True if editing software was found
            - ``software``  : str  — raw value of the Software tag (or empty)
            - ``suspicious``: bool — True if a known editing keyword matched
        """
        software = self.get_software() or ""
        lower = software.lower()
        suspicious = any(kw in lower for kw in _EDITING_SOFTWARE_KEYWORDS)
        return {
            "detected": bool(software),
            "software": software,
            "suspicious": suspicious,
        }

    def compute_manipulation_score(self) -> float:
        """
        Heuristic score (0 – 1) estimating the likelihood of manipulation
        based on metadata alone.

        Scoring rubric:
        +0.3  — EXIF is completely absent (common for synthetic / cropped images)
        +0.3  — Editing software detected in EXIF
        +0.2  — No camera make/model present
        +0.2  — No creation date present

        Returns:
            Float in [0.0, 1.0] where higher means more suspicious.
        """
        self._load()
        score = 0.0

        if not self.has_exif():
            score += 0.3
        else:
            editing = self.detect_editing_software()
            if editing["suspicious"]:
                score += 0.3

            cam = self.get_camera_make_model()
            if not cam["make"] and not cam["model"]:
                score += 0.2

            if not self.get_creation_date():
                score += 0.2

        return min(score, 1.0)

    def summary(self) -> dict:
        """
        Return a compact summary of all metadata analysis results.

        Returns:
            Dictionary with the following keys:
            ``has_exif``, ``creation_date``, ``camera``, ``software``,
            ``editing_detected``, ``manipulation_score``.
        """
        self._load()
        cam = self.get_camera_make_model()
        editing = self.detect_editing_software()
        return {
            "has_exif": self.has_exif(),
            "creation_date": self.get_creation_date(),
            "camera": f"{cam['make'] or '?'} {cam['model'] or '?'}".strip(),
            "software": editing["software"] or "N/A",
            "editing_detected": editing["suspicious"],
            "manipulation_score": self.compute_manipulation_score(),
        }
