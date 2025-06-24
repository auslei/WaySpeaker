import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

os.environ["TRANSFORMERS_OFFLINE"] = "1"

def main():
    # Set device (use GPU if available, otherwise CPU)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Paths
    checkpoints_path = 'checkpoints/checkpoints_v2'
    ckpt_converter = f'{checkpoints_path}/converter'
    output_dir = 'output'  # Make sure this matches your actual folder name
    os.makedirs(output_dir, exist_ok=True)  # Create output dir if it doesn't exist

    # Initialize MeloTTS model for Chinese
    model = TTS(language="ZH", device=device)
    print("Available speakers:", list(model.hps.data.spk2id.keys()))
    speaker_ids = model.hps.data.spk2id

    # Initialize tone color converter with its config and checkpoint
    tone_color_converter = ToneColorConverter(
        config_path=f'{ckpt_converter}/config.json',
        device=device
    )
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    # Load speaker embedding (SE) from a reference voice sample
    reference_speaker = 'reference_voice/leilei_cn2.m4a'
    target_se, _ = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

    # Texts to synthesize (can add more languages/voices as needed)
    texts = {
        'ZH': (
            "欢迎来到位于Mt Waverley的Amber Grove——现代建筑与永恒舒适的完美结合。"
            "这座精心设计的住宅拥有宽敞的开放式格局、高端装修，"
            "以及室内外自然衔接的生活空间。坐落在宁静绿荫街道，"
            "是追求品质、空间与便利的家庭理想之选。"
        )
    }

    # Temporary output path for base synthesis before tone conversion
    tmp_path = os.path.join(output_dir, 'tmp.wav')
    speed = 0.8  # Synthesis speed multiplier

    for language, text in texts.items():
        # Get speaker ID for the TTS model
        speaker_id = speaker_ids[language]

        # Load the base speaker embedding used for TTS synthesis
        speaker_key = language.lower().replace('_', '-')
        source_se_path = f'{checkpoints_path}/base_speakers/ses/{speaker_key}.pth'
        source_se = torch.load(source_se_path, map_location=device)

        # Synthesize base audio from text using source speaker
        model.tts_to_file(text, speaker_id, tmp_path, speed=speed)

        # Output path for final converted voice
        final_path = os.path.join(output_dir, f'output_v2_{language}.wav')

        # Apply tone color conversion to make the voice sound like the reference speaker
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=tmp_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=final_path,
            message=encode_message
        )


if __name__ == "__main__":
    main()
