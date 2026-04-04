# Copyright 2026 Tencent.
from typing import Any

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.covo_audio.config_covo_audio import COVO_AUDIO_TOKEN_INDEX


def llm2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt=None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    talker_outputs = stage_list[source_stage_id].engine_outputs
    code2wav_inputs = []

    for i, talker_output in enumerate(talker_outputs):
        output = talker_output.outputs[0]
        token_ids = output.token_ids

        audio_codes = [t - COVO_AUDIO_TOKEN_INDEX for t in token_ids if t >= COVO_AUDIO_TOKEN_INDEX]

        if not audio_codes:
            audio_codes = [-1]

        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=audio_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
