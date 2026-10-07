"""Download the HockeyAI model weights from Hugging Face."""

from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "SimulaMet-HOST/HockeyAI"
FILENAME = "HockeyAI_model_weight.pt"


def main() -> None:
    target_dir = Path("models")
    target_dir.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=target_dir,
    )
    print(f"HockeyAI weights available at: {path}")


if __name__ == "__main__":
    main()
