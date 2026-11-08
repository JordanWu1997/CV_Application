"""Constant quality MP4 output shared by the face video pipelines."""

import shutil
import subprocess
import tempfile

import numpy as np


def add_encoding_arguments(parser):
    parser.add_argument('--video-codec', choices=('h264', 'hevc'), default='h264',
                        help='MP4 codec (default: h264)')
    parser.add_argument('--video-crf', type=int, default=28,
                        help='Constant quality, 0-51; higher is smaller (default: 28)')
    parser.add_argument('--video-preset', choices=EncodedVideoWriter.PRESETS,
                        default='medium',
                        help='Encoding speed; slower usually produces smaller files')


class EncodedVideoWriter:
    """Write BGR uint8 frames to FFmpeg using H.264 or H.265."""

    ENCODERS = {'h264': 'libx264', 'hevc': 'libx265'}
    PRESETS = ('ultrafast', 'superfast', 'veryfast', 'faster', 'fast',
               'medium', 'slow', 'slower', 'veryslow')

    def __init__(self, path, size, fps, codec='h264', crf=28, preset='medium'):
        if not shutil.which('ffmpeg'):
            raise RuntimeError('FFmpeg is required to encode output video.')
        if codec not in self.ENCODERS:
            raise ValueError(f'Unsupported video codec: {codec}')
        if preset not in self.PRESETS:
            raise ValueError(f'Unsupported encoder preset: {preset}')
        if not 0 <= crf <= 51:
            raise ValueError('CRF must be between 0 and 51.')
        self.width, self.height = map(int, size)
        if self.width <= 0 or self.height <= 0 or fps <= 0:
            raise ValueError('Video size and FPS must be positive.')
        self.path = str(path)
        self.stderr = tempfile.TemporaryFile(mode='w+t')
        command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}', '-r', str(fps),
            '-i', 'pipe:0', '-an', '-c:v', self.ENCODERS[codec],
            '-preset', preset, '-crf', str(crf),
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', self.path,
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                        stderr=self.stderr)

    def write(self, frame):
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(f'Expected BGR uint8 frame of shape '
                             f'({self.height}, {self.width}, 3); got '
                             f'{frame.shape} {frame.dtype}')
        if self.process.stdin is None:
            raise RuntimeError('Video writer is closed.')
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as error:
            self.process.wait()
            raise RuntimeError(self._error_text()) from error

    def _error_text(self):
        self.stderr.seek(0)
        detail = self.stderr.read().strip()
        return f'FFmpeg failed to write {self.path}: {detail or "unknown error"}'

    def release(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None
        code = self.process.wait()
        if code != 0:
            message = self._error_text()
            self.stderr.close()
            raise RuntimeError(message)
        self.stderr.close()
