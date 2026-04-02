"""
E2E tests for Covo-Audio-Chat model with audio input and audio/text output.
"""

import shutil
from pathlib import Path

import pytest

from tests.conftest import generate_synthetic_audio, modify_stage_config
from tests.utils import hardware_test

models = ["tencent/Covo-Audio-Chat"]

COVO_AUDIO_SYSTEM_PROMPT = (
    '你是"小腾"，英文名是"Covo"，由腾讯开发的AI助手。\n'
    "1、请使用简洁、口语化的语言和用户聊天，"
    "你的态度积极、耐心，像一位值得信赖的朋友。\n"
    "2、不要使用列表或编号，避免输出网址、表情符号和复杂的公式。\n"
    "3、不评价竞争对手，不发表主观政治观点，"
    "针对色情类、政治类、恐怖类、歧视类、暴力类的用户问题，"
    "你要妥善应对潜在的安全风险，并给出幽默，情绪安抚以及安全的劝导。\n"
    "请用文本和音频进行对话，交替生成5个文本token和15个音频token，"
    "音频部分使用发音人：default_female"
)


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


def _build_covo_prompt(
    prompt_text: str,
    *,
    has_audio: bool = False,
    system_prompt: str = COVO_AUDIO_SYSTEM_PROMPT,
) -> str:
    """Build a chat-template prompt for Covo-Audio-Chat."""
    user_content = ""
    if has_audio:
        user_content += "<|begofcAUDIO|><|cAUDIO|><|endofcAUDIO|>"
    user_content += prompt_text

    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


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

    prompt = _build_covo_prompt(get_question("audio_chat"), has_audio=True)
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
    prompt = _build_covo_prompt(get_question("text_only"))
    omni_inputs = [{"prompt": prompt, "modalities": ["audio"]}]

    outputs = omni_runner.generate(omni_inputs)
    assert len(outputs) > 0, "Expected at least one output"
