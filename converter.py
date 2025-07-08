"""
Convert base TTS audio into a target speaker's tone/color using openvoice.

Usage:
  python converter.py <text> <language> <reference_file>
                      [--output-dir DIR] [--speed FLOAT]
                      [--encode-message TEXT]
"""
import os
import argparse
import logging
import tempfile
from pathlib import Path

# Ensure offline mode for transformers before any imports that may load it
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

# module-level logger
logger = logging.getLogger(__name__)


def get_device() -> str:
    """Auto-detect torch device: MPS (Apple M1), CUDA, or CPU."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda:0'

    return 'cpu'

# Default checkpoints directory (v2)
CKPT_DIR = Path('checkpoints/checkpoints_v2')


def read_text_input(input_text: str) -> str:
    """Read text from input which can be either a string or a file path."""
    if os.path.isfile(input_text):
        try:
            with open(input_text, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise ValueError(f"Error reading file {input_text}: {str(e)}")
    return input_text


def convert_text(
    text: str,
    speaker_key: str,
    reference_file: Path,
    output_dir: Path,
    speed: float = 0.8,
    encode_message: str = '@MyShell', #add a watermark to identify origin of the sound.
) -> Path:
    """
    Synthesize `text` for `speaker_key` and apply tone-color conversion using the
    reference voice in `reference_file`. Results are written to `output_dir`.

    Returns the path to the converted WAV file.
    """
    device = get_device()
    # Initialize TTS model
    model = TTS(language=speaker_key, device=device)
    # Load base speaker embedding
    file_key = speaker_key.lower().replace('_', '-')
    source_se = torch.load(
        str(CKPT_DIR / 'base_speakers' / 'ses' / f'{file_key}.pth'),
        map_location=device,
    )
    # Initialize converter
    converter = ToneColorConverter(
        config_path=str(CKPT_DIR / 'converter' / 'config.json'),
        device=device,
    )
    converter.load_ckpt(str(CKPT_DIR / 'converter' / 'checkpoint.pth'))
    # Extract target speaker embedding
    target_se, _ = se_extractor.get_se(str(reference_file), converter, vad=True)

    # Synthesize base audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        tmp_path = Path(tmp_wav.name)
    with torch.no_grad():
        speaker_id = model.hps.data.spk2id[speaker_key]
        model.tts_to_file(text, speaker_id, str(tmp_path), speed=speed)

    # Apply tone-color conversion
    final_path = output_dir / f'output_{speaker_key}.wav'
    with torch.no_grad():
        converter.convert(
            audio_src_path=str(tmp_path),
            src_se=source_se,
            tgt_se=target_se,
            output_path=str(final_path),
            message=encode_message,
        )
    try:
        tmp_path.unlink()
    except Exception:
        pass
    logger.info('Generated converted audio: %s', final_path)
    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize text and apply tone-color conversion"
    )
    parser.add_argument('text', help='Text to synthesize (either direct string or path to a text file)')
    parser.add_argument(
        'language',
        help='Speaker key/language (e.g. ZH, JA, EN_US, EN_UK, EN_AU)',
    )
    parser.add_argument(
        'reference_file',
        type=Path,
        help='Path to reference audio file for target voice embedding',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output'),
        help='Directory to write converted audio',
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=0.8,
        help='Synthesis speed multiplier',
    )
    parser.add_argument(
        '--encode-message',
        type=str,
        default='@MyShell',
        help='Message tag for tone-color encoder',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    convert_text(
        text=read_text_input(args.text),
        speaker_key=args.language,
        reference_file=args.reference_file,
        output_dir=args.output_dir,
        speed=args.speed,
        encode_message=args.encode_message,
    )


if __name__ == '__main__':
    main()
