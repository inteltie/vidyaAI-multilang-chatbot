"""LangGraph node: save_memory."""

from __future__ import annotations
import logging
import asyncio
from time import perf_counter
from typing import Any, Dict, List

from services import MemoryService
from services.message_modality import build_assistant_modality, build_user_modality
from state import AgentState

logger = logging.getLogger(__name__)


class SaveMemoryNode:
    """LangGraph node: save_memory."""

    def __init__(self, memory_service: MemoryService) -> None:
        self._memory_service = memory_service
    
    async def _save_messages_sequentially(
        self,
        session_id: str,
        user_id: str,
        query: str,
        response: str,
        user_modality: dict,
        assistant_modality: dict,
    ):
        """Save user message first, then AI message to ensure correct timestamp order."""
        await self._memory_service.background_save_message(
            session_id,
            user_id,
            "user",
            query,
            has_image_input=user_modality.get("has_image_input", False),
            image_input_ids=user_modality.get("image_input_ids"),
            image_input_urls=user_modality.get("image_input_urls"),
            has_generated_image=False,
        )
        await self._memory_service.background_save_message(
            session_id,
            user_id,
            "assistant",
            response,
            has_generated_image=assistant_modality.get("has_generated_image", False),
            generated_image_ids=assistant_modality.get("generated_image_ids"),
            generated_image_urls=assistant_modality.get("generated_image_urls"),
        )

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        start = perf_counter()
        user_session_id = state["user_session_id"]
        user_id = state["user_id"]
        query = state.get("effective_query") or state.get("query", "")
        response = state.get("response", "")

        logger.info(f"[DEBUG_SAVE] Session: {user_session_id} | Query: {(query or '')[:50]} | Resp: {(response or '')[:50]}...")

        # Safety check: Don't save if response is empty (prevents half-saves)
        if not response or not query:
            logger.warning("Attempted to save empty query or response. Skipping memory update.")
            return {"timings": {"save_memory": 0}}

        try:
            if state.get("new_uploaded_image_records"):
                await self._memory_service.append_uploaded_images(
                    user_session_id,
                    user_id,
                    state.get("new_uploaded_image_records") or [],
                )
        except Exception as exc:
            logger.warning("Failed to persist uploaded image metadata: %s", exc)

        try:
            if state.get("new_generated_image_records"):
                await self._memory_service.append_generated_images(
                    user_session_id,
                    user_id,
                    state.get("new_generated_image_records") or [],
                )
        except Exception as exc:
            logger.warning("Failed to persist generated image metadata: %s", exc)

        try:
            preferred_format = state.get("preferred_response_format")
            if preferred_format:
                await self._memory_service.update_user_preferences(
                    user_session_id,
                    user_id,
                    response_format=str(preferred_format),
                )
        except Exception as exc:
            logger.warning("Failed to persist user preferences: %s", exc)

        user_modality = build_user_modality(state)
        assistant_modality = build_assistant_modality(state)

        # 1. Update Redis buffer (await - fast)
        await self._memory_service.add_message(
            user_session_id,
            "user",
            query,
            has_image_input=user_modality["has_image_input"],
            image_input_ids=user_modality["image_input_ids"],
            image_input_urls=user_modality["image_input_urls"],
        )
        await self._memory_service.add_message(
            user_session_id,
            "assistant",
            response,
            has_generated_image=assistant_modality["has_generated_image"],
            generated_image_ids=assistant_modality["generated_image_ids"],
            generated_image_urls=assistant_modality["generated_image_urls"],
        )

        # 2. Save to MongoDB sequentially (background)
        asyncio.create_task(
            self._save_messages_sequentially(
                user_session_id,
                user_id,
                query,
                response,
                user_modality,
                assistant_modality,
            )
        )

        duration = perf_counter() - start
        return {
            "timings": {"save_memory": duration},
            "documents": []  # Clear documents to prevent stale reuse in next turn
        }


