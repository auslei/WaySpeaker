"""
WaySpeaker - Voice Cloning Tool

Convert text to speech with voice cloning using OpenVoice and MeloTTS.
This tool synthesizes text in a base speaker's voice and then applies
tone-color conversion to match a reference voice.

Usage:
  python converter.py <text> <language> <reference_file>
                      [--output-dir DIR] [--speed FLOAT]
                      [--encode-message TEXT]

Example:
  python converter.py "Hello world" EN_US reference.wav --speed 1.0
"""
import os
import argparse
import logging
import tempfile
from pathlib import Path

# Enable online mode for transformers to download models if needed
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
    """
    Read text from input which can be either a direct string or a file path.
    
    Args:
        input_text: Either direct text to synthesize or path to a text file
        
    Returns:
        The text content to be synthesized
        
    Raises:
        ValueError: If the file cannot be read
    """
    if os.path.isfile(input_text):
        try:
            with open(input_text, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    raise ValueError(f"File {input_text} is empty")
                return content
        except (OSError, IOError) as e:
            raise ValueError(f"Error reading file {input_text}: {str(e)}")
    return input_text


def convert_text(
    text: str,
    speaker_key: str,
    reference_file: Path,
    output_dir: Path,
    speed: float = 0.8,
    encode_message: str = '@MyShell',
) -> Path:
    """
    Synthesize text and apply tone-color conversion using a reference voice.
    
    This function performs two main steps:
    1. Synthesizes text using MeloTTS in a base speaker's voice
    2. Applies tone-color conversion to match the reference voice
    
    Args:
        text: Text to synthesize
        speaker_key: Language/speaker code (e.g., 'ZH', 'EN_US', 'JA')
        reference_file: Path to reference audio file for voice cloning
        output_dir: Directory to write the converted audio file
        speed: Speech synthesis speed multiplier (default: 0.8)
        encode_message: Watermark message embedded in the audio (default: '@MyShell')
        
    Returns:
        Path to the generated WAV file
        
    Raises:
        FileNotFoundError: If reference file or checkpoints are missing
        ValueError: If speaker_key is not supported
    """
    # Validate inputs
    if not reference_file.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_file}")
    
    device = get_device()
    logger.info('Using device: %s', device)
    
    # Initialize TTS model
    try:
        model = TTS(language=speaker_key, device=device)
    except Exception as e:
        raise ValueError(f"Invalid speaker key '{speaker_key}': {str(e)}")
    
    # Load base speaker embedding
    file_key = speaker_key.lower().replace('_', '-')
    speaker_embedding_path = CKPT_DIR / 'base_speakers' / 'ses' / f'{file_key}.pth'
    
    if not speaker_embedding_path.exists():
        raise FileNotFoundError(
            f"Speaker embedding not found: {speaker_embedding_path}. "
            f"Please ensure checkpoints are properly installed."
        )
    
    source_se = torch.load(str(speaker_embedding_path), map_location=device)
    # Initialize converter
    converter_config_path = CKPT_DIR / 'converter' / 'config.json'
    converter_ckpt_path = CKPT_DIR / 'converter' / 'checkpoint.pth'
    
    if not converter_config_path.exists() or not converter_ckpt_path.exists():
        raise FileNotFoundError(
            f"Converter checkpoints not found in {CKPT_DIR / 'converter'}. "
            f"Please ensure checkpoints are properly installed."
        )
    
    converter = ToneColorConverter(
        config_path=str(converter_config_path),
        device=device,
    )
    converter.load_ckpt(str(converter_ckpt_path))
    
    # Extract target speaker embedding
    logger.info('Extracting voice embedding from reference file...')
    target_se, _ = se_extractor.get_se(str(reference_file), converter, vad=True)

    # Synthesize base audio
    logger.info('Synthesizing base audio...')
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        tmp_path = Path(tmp_wav.name)
    
    try:
        with torch.no_grad():
            speaker_id = model.hps.data.spk2id[speaker_key]
            model.tts_to_file(text, speaker_id, str(tmp_path), speed=speed)
    except KeyError:
        raise ValueError(
            f"Speaker key '{speaker_key}' not found in model. "
            f"Available keys: {list(model.hps.data.spk2id.keys())}"
        )

    # Apply tone-color conversion
    logger.info('Applying voice conversion...')
    final_path = output_dir / f'output_{speaker_key}.wav'
    
    try:
        with torch.no_grad():
            converter.convert(
                audio_src_path=str(tmp_path),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(final_path),
                message=encode_message,
            )
    finally:
        # Clean up temporary file
        try:
            tmp_path.unlink()
        except Exception as e:
            logger.warning('Failed to delete temporary file %s: %s', tmp_path, e)
    
    logger.info('Successfully generated audio: %s', final_path)
    return final_path


def main() -> None:
    """Main entry point for the WaySpeaker voice cloning tool."""
    parser = argparse.ArgumentParser(
        description="WaySpeaker - Text-to-Speech with Voice Cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python converter.py "Hello world" EN_US reference.wav
  
  # From file with custom speed
  python converter.py input.txt ZH reference.wav --speed 1.0
  
  # Custom output directory
  python converter.py "Text" EN_US ref.wav --output-dir ./my_output
        """
    )
    parser.add_argument(
        'text',
        help='Text to synthesize (either direct string or path to a text file)'
    )
    parser.add_argument(
        'language',
        help='Speaker key/language code (e.g., ZH, JA, EN_US, EN_UK, EN_AU)',
    )
    parser.add_argument(
        'reference_file',
        type=Path,
        help='Path to reference audio file for voice cloning',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output'),
        help='Directory to write converted audio (default: output)',
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=0.8,
        help='Speech synthesis speed multiplier (default: 0.8)',
    )
    parser.add_argument(
        '--encode-message',
        type=str,
        default='@MyShell',
        help='Watermark message embedded in the audio (default: @MyShell)',
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read and validate text input
        text = read_text_input(args.text)
        if not text:
            raise ValueError("Text input is empty")
        
        logger.info('Starting voice conversion...')
        logger.info('Text length: %d characters', len(text))
        logger.info('Language: %s', args.language)
        logger.info('Reference file: %s', args.reference_file)
        
        # Perform conversion
        output_path = convert_text(
            text=text,
            speaker_key=args.language,
            reference_file=args.reference_file,
            output_dir=args.output_dir,
            speed=args.speed,
            encode_message=args.encode_message,
        )
        
        logger.info('Conversion complete! Output saved to: %s', output_path)
        
    except (ValueError, FileNotFoundError) as e:
        logger.error('Error: %s', e)
        raise SystemExit(1)
    except KeyboardInterrupt:
        logger.info('Conversion cancelled by user')
        raise SystemExit(130)
    except Exception as e:
        logger.error('Unexpected error: %s', e, exc_info=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
