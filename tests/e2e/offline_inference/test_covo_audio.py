"""
E2E tests for Covo-Audio-Chat model with audio input and audio/text output.
"""

from pathlib import Path

import librosa
import numpy as np
import pytest

from tests.conftest import modify_stage_config
from tests.utils import hardware_test
from vllm_omni.model_executor.models.covo_audio.prompt_utils import (
    COVO_AUDIO_INPUT_PREFIX,
    build_covo_audio_chat_prompt,
)

SAMPLE_AUDIO_PATH = str(
    Path(__file__).parent.parent.parent.parent / "examples" / "offline_inference" / "covo_audio" / "sample_audio.wav"
)

models = ["tencent/Covo-Audio-Chat"]


def get_eager_config():
    return modify_stage_config(
        str(Path(__file__).parent.parent / "stage_configs" / "covo_audio_ci.yaml"),
        updates={
            "stage_args": {
                0: {"engine_args.enforce_eager": "true"},
                1: {"engine_args.enforce_eager": "true"},
            },
        },
    )


stage_config = get_eager_config()
test_params = [(model, stage_config) for model in models]


def get_question():
    return "请回答这段音频里的问题。"


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_audio_to_audio(omni_runner, omni_runner_handler) -> None:
    """
    Test audio input → text + audio output.
    Deploy Setting: default yaml
    Input Modal: text + audio
    Output Modal: audio
    """
    audio, _ = librosa.load(SAMPLE_AUDIO_PATH, sr=16000)
    audio = audio.astype(np.float32)

    user_content = COVO_AUDIO_INPUT_PREFIX + get_question()
    prompt = build_covo_audio_chat_prompt(user_content)
    omni_inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"audio": (audio, 16000)},
            "modalities": ["audio"],
        }
    ]

    outputs = omni_runner.generate(omni_inputs)
    assert len(outputs) > 0, "Expected at least one output"
