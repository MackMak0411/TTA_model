from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

from train import get_audio_encoder, get_dataset, get_model, get_tokenizer
from TTA.utils import parse_yaml


CLAP_SAMPLE_RATE = 48000


@torch.inference_mode()
def clap_score(
    processor: ClapProcessor,
    clap_model: ClapModel,
    audio: torch.Tensor,
    caption: str,
    sample_rate: int,
    device: str,
) -> float:
    """Return cosine similarity between generated audio and its caption."""

    audio = audio.detach().float().cpu().flatten()
    if sample_rate != CLAP_SAMPLE_RATE:
        audio = torchaudio.functional.resample(
            audio, sample_rate, CLAP_SAMPLE_RATE
        )

    audio_inputs = processor(
        audios=audio.numpy(),
        sampling_rate=CLAP_SAMPLE_RATE,
        return_tensors="pt",
    )
    text_inputs = processor(
        text=[caption],
        return_tensors="pt",
        padding=True,
    )
    audio_inputs = {
        key: value.to(device) for key, value in audio_inputs.items()
    }
    text_inputs = {
        key: value.to(device) for key, value in text_inputs.items()
    }

    audio_features = clap_model.get_audio_features(**audio_inputs)
    text_features = clap_model.get_text_features(**text_inputs)
    audio_features = F.normalize(audio_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    return float((audio_features * text_features).sum(dim=-1).item())


@torch.inference_mode()
def generate_audio(
    caption: str,
    configs: dict,
    audio_encoder: torch.nn.Module,
    tokenizer: object,
    model: torch.nn.Module,
    device: str,
    mixed_precision_enabled: bool,
) -> torch.Tensor:
    """Generate one audio waveform from a caption."""

    caption_ids = tokenizer.captions_to_ids(
        captions=[caption],
        fix_length=configs["max_caption_len"],
    ).to(device)
    audio_ids = torch.full(
        (1, 1),
        tokenizer.boa_token_id,
        dtype=torch.long,
        device=device,
    )
    input_ids = torch.cat((caption_ids, audio_ids), dim=1)
    max_new_ids = max(1, model.block_size - input_ids.shape[1])
    # Match training precision to reduce GPU memory during generation.
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=mixed_precision_enabled,
    ):
        ids = model.generate(
        ids=input_ids,
        max_new_ids=max_new_ids,
        temperature=1.0,
        top_k=200,
        stop_token_id=tokenizer.eoa_token_id,
        )
    audio_codes = tokenizer.ids_to_audio_codes(ids)
    if audio_codes.shape[1] == 0:
        raise RuntimeError(f"No audio codes were generated for: {caption!r}")

    audio = audio_encoder.decode(audio_codes)[0]
    peak = audio.abs().amax()
    if peak < 1e-6:
        raise RuntimeError(f"Generated silent audio for: {caption!r}")
    return audio / peak


def evaluate(args) -> None:
    configs = parse_yaml(args.config)
    device = configs["train"]["device"]
    clap_device = "cpu"
    mixed_precision_enabled = (
        configs["train"].get("mixed_precision", False)
        and device.startswith("cuda")
    )

    audio_encoder = get_audio_encoder(configs).to(device)
    tokenizer = get_tokenizer(configs, audio_encoder)
    model = get_model(
        configs=configs,
        vocab_size=len(tokenizer),
        ckpt_path=args.ckpt_path,
    ).to(device)
    model.eval()

    evaluation_dataset = get_dataset(configs, split="evaluation")
    clap_model_name = "laion/clap-htsat-unfused"
    processor = ClapProcessor.from_pretrained(clap_model_name)
    clap_model = ClapModel.from_pretrained(clap_model_name).to(clap_device)
    clap_model.eval()

    generated_scores = []
    real_scores = []
    sample_count = min(args.max_samples, len(evaluation_dataset))
    for index in tqdm(range(sample_count), desc="Evaluating"):
        sample = evaluation_dataset[index]
        caption = sample["caption"]
        real_score = clap_score(
            processor=processor,
            clap_model=clap_model,
            audio=sample["audio"],
            caption=caption,
            sample_rate=configs["sample_rate"],
            device=clap_device,
        )
        audio = generate_audio(
            caption=caption,
            configs=configs,
            audio_encoder=audio_encoder,
            tokenizer=tokenizer,
            model=model,
            device=device,
            mixed_precision_enabled=mixed_precision_enabled,
        )
        score = clap_score(
            processor=processor,
            clap_model=clap_model,
            audio=audio,
            caption=caption,
            sample_rate=configs["sample_rate"],
            device=clap_device,
        )
        real_scores.append(real_score)
        generated_scores.append(score)

    if not generated_scores:
        raise RuntimeError("The evaluation dataset contains no samples.")

    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Samples: {len(generated_scores)}")
    print("Real audio CLAP mean cosine similarity: "f"{np.mean(real_scores):.4f}")
    print("Real audio CLAP standard deviation: "f"{np.std(real_scores):.4f}")
    print("Generated audio CLAP mean cosine similarity: "f"{np.mean(generated_scores):.4f}")
    print("Generated audio CLAP standard deviation: "f"{np.std(generated_scores):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, required=True)
    args = parser.parse_args()
    evaluate(args)

