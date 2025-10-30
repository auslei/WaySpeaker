# WaySpeaker

Text-to-Speech voice cloning tool that converts synthesized speech into a target speaker's tone and vocal characteristics using OpenVoice and MeloTTS.

## Features

- 🎤 **Voice Cloning**: Convert any text into speech matching a reference voice
- 🌍 **Multi-language Support**: Supports Chinese (ZH), Japanese (JA), English (US, UK, AU), and more
- 🎛️ **Customizable Speed**: Adjust speech synthesis speed
- 📁 **Flexible Input**: Accept text directly or from a file
- 🔊 **High Quality**: Powered by OpenVoice v2 and MeloTTS

## Prerequisites

- Python 3.8 or higher
- PyTorch (with CUDA support for GPU acceleration, or MPS for Apple Silicon)
- OpenVoice v2 checkpoints (download required)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/auslei/WaySpeaker.git
cd WaySpeaker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download OpenVoice v2 checkpoints:
   - Create a `checkpoints/checkpoints_v2` directory in the project root
   - Download the model files from [OpenVoice GitHub](https://github.com/myshell-ai/OpenVoice)
   - Place the files in the checkpoint directory with the following structure:
     ```
     checkpoints/checkpoints_v2/
     ├── base_speakers/
     │   └── ses/
     │       ├── zh.pth
     │       ├── en-us.pth
     │       └── ...
     └── converter/
         ├── config.json
         └── checkpoint.pth
     ```

## Usage

### Basic Usage

```bash
python converter.py "Hello, this is a test" EN_US reference_voice/sample.wav
```

### Read from a File

```bash
python converter.py input.txt EN_US reference_voice/sample.wav
```

### Advanced Options

```bash
python converter.py "Your text here" ZH reference_voice/sample.wav \
    --output-dir ./my_output \
    --speed 1.0 \
    --encode-message "@CustomWatermark"
```

### Command Line Arguments

- `text` (required): Text to synthesize or path to a text file
- `language` (required): Speaker key/language code
  - Supported: `ZH` (Chinese), `JA` (Japanese), `EN_US`, `EN_UK`, `EN_AU`, `EN_DEFAULT`, `ES`, `FR`, `KR`, `ZH_MIX_EN`
- `reference_file` (required): Path to reference audio file (WAV, M4A, MP3, etc.)
- `--output-dir`: Directory for output files (default: `./output`)
- `--speed`: Speech synthesis speed multiplier (default: 0.8)
- `--encode-message`: Watermark message for the audio (default: `@MyShell`)

## Examples

### Chinese Voice Cloning
```bash
python converter.py "你好，世界" ZH reference_voice/chinese_speaker.wav
```

### English Voice Cloning with Custom Speed
```bash
python converter.py "Welcome to WaySpeaker" EN_US reference_voice/english_speaker.wav --speed 1.2
```

### Long Text from File
```bash
# Create a text file
echo "This is a longer text that I want to convert to speech" > story.txt

# Convert it
python converter.py story.txt EN_US reference_voice/narrator.wav
```

## Project Structure

```
WaySpeaker/
├── converter.py          # Main conversion script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── reference_voice/     # Sample reference voice files
├── output/              # Generated audio files
└── checkpoints/         # Model checkpoints (not included)
```

## How It Works

1. **Text Synthesis**: MeloTTS synthesizes the input text in the base speaker's voice
2. **Voice Extraction**: Extracts voice embeddings from your reference audio file
3. **Tone Conversion**: OpenVoice's tone-color converter applies the reference voice characteristics to the synthesized speech
4. **Output**: Generates a WAV file with the target voice speaking the input text

## Supported Languages

The following language codes are supported (based on MeloTTS):
- `ZH` - Chinese (Mandarin)
- `EN_US` - English (US)
- `EN_UK` - English (UK)
- `EN_AU` - English (Australian)
- `EN_DEFAULT` - English (Default)
- `JA` - Japanese
- `ES` - Spanish
- `FR` - French
- `KR` - Korean
- `ZH_MIX_EN` - Chinese mixed with English

## Hardware Requirements

- **CPU**: Works on any modern CPU (slower processing)
- **GPU**: NVIDIA GPU with CUDA support recommended for faster processing
- **Apple Silicon**: Supports MPS acceleration on M1/M2/M3 Macs
- **Memory**: At least 4GB RAM recommended

## Troubleshooting

### Common Issues

**Import errors**: Make sure all dependencies are installed
```bash
pip install -r requirements.txt --upgrade
```

**Checkpoint not found**: Verify the checkpoint directory structure matches the expected format

**CUDA/MPS errors**: The script auto-detects available devices. If GPU acceleration fails, it falls back to CPU

**Audio quality issues**: Try adjusting the `--speed` parameter or use a higher quality reference audio file

## Credits

This project is built on top of:
- [OpenVoice](https://github.com/myshell-ai/OpenVoice) - Voice tone-color cloning
- [MeloTTS](https://github.com/myshell-ai/MeloTTS) - Text-to-speech synthesis
- [MyShell.ai](https://myshell.ai/) - Model providers

## License

This project follows the licenses of its dependencies. Please refer to the original repositories for license information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
