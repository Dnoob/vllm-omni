"""
E2E tests for Covo-Audio-Chat model with audio input and audio/text output.
"""

from pathlib import Path

import pytest

from tests.conftest import generate_synthetic_audio, modify_stage_config
from tests.utils import hardware_test
from vllm_omni.model_executor.models.covo_audio.prompt_utils import (
    COVO_AUDIO_INPUT_PREFIX,
    build_covo_audio_chat_prompt,
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


@pytest.mark.advanced_model
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
    audio = generate_synthetic_audio(2, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()

    user_content = COVO_AUDIO_INPUT_PREFIX + get_question()
    prompt = build_covo_audio_chat_prompt(user_content)

    request_config = {
        "prompts": prompt,
        "audios": (audio, 16000),
        "modalities": ["audio"],
    }
    omni_runner_handler.send_request(request_config)
