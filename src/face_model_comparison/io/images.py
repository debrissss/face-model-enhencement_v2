"""PNG 序列化只负责图片读写，不包含 Stage 分发。"""
from pathlib import Path
from PIL import Image


class ImageSerializer:
    def save(self, data: Image.Image, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.save(path, format="PNG")

    def load(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB").copy()


SERIALIZERS = {Image.Image: ImageSerializer()}


class ArtifactStore:
    def save(self, data: Image.Image, path: Path) -> Path:
        SERIALIZERS[type(data)].save(data, path)
        return path

    def load(self, path: Path) -> Image.Image:
        return SERIALIZERS[Image.Image].load(path)
