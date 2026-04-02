"""
E2E tests for Covo-Audio-Chat model with audio input and audio/text output.
"""

import shutil
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


def get_question(prompt_type="audio_chat"):
    prompts = {
        "audio_chat": "请回答这段音频里的问题。",
        "text_only": "你好，请介绍一下你自己。",
    }
    return prompts.get(prompt_type, prompts["audio_chat"])


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_audio_to_audio(omni_runner, omni_runner_handler) -> None:
    """
    Test audio input → text + audio output.
    Deploy Setting: default yaml
    Input Modal: text + audio
    Output Modal: audio
    """
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()

    user_content = COVO_AUDIO_INPUT_PREFIX + get_question("audio_chat")
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


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_audio(omni_runner, omni_runner_handler) -> None:
    """
    Test text-only input → text + audio output.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: audio
    """
    prompt = build_covo_audio_chat_prompt(get_question("text_only"))
    omni_inputs = [{"prompt": prompt, "modalities": ["audio"]}]

    outputs = omni_runner.generate(omni_inputs)
    assert len(outputs) > 0, "Expected at least one output"
