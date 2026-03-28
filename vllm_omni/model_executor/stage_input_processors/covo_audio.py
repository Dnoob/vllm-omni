# Copyright 2026 Tencent.
from typing import Any

from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.covo_audio.config_covo_audio import COVO_AUDIO_TOKEN_INDEX

logger = init_logger(__name__)


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
        request_id = getattr(talker_output, "request_id", f"unknown_{i}")
        logger.debug(
            "Request %s: total_tokens=%d, text_tokens=%d, audio_tokens=%d",
            request_id,
            len(token_ids),
            len([t for t in token_ids if t < COVO_AUDIO_TOKEN_INDEX]),
            len(audio_codes),
        )

        if not audio_codes:
            logger.info(
                "Request %s: no audio codes in Stage-0 output "
                "(text-only response). Sending sentinel [-1] to Stage-1; "
                "code2wav will return silence.",
                request_id,
            )
            audio_codes = [-1]

        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=audio_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
