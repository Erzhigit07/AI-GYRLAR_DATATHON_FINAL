"""Regenerate tiny synthetic MOV fixtures for the browser's aspect-ratio checks.

Requires the project's existing PyAV and NumPy dependencies; no private video.
"""

from pathlib import Path

import av
import numpy as np


def main() -> None:
    for width, height in [(360, 640), (640, 360), (480, 480)]:
        output = Path(__file__).with_name(f"synthetic-{width}x{height}.mov")
        pixels = np.full((height, width, 3), (233, 216, 225), dtype=np.uint8)
        pixels[10:-10, 10:-10] = (135, 35, 77)
        pixels[18:-18, 18:-18] = (233, 216, 225)
        pixels[height // 2 - 4 : height // 2 + 4, 18:-18] = (135, 35, 77)
        pixels[18:-18, width // 2 - 4 : width // 2 + 4] = (135, 35, 77)
        with av.open(str(output), mode="w", format="mov") as container:
            stream = container.add_stream("libx264", rate=10)
            stream.width, stream.height = width, height
            stream.pix_fmt = "yuv420p"
            for _ in range(5):
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)


if __name__ == "__main__":
    main()
