"""Personas: saved system prompt + answer-length tuning + model choice."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .llm.base import GenerationConfig
from .storage import Database, new_id

LengthPreset = Literal["short", "normal", "long", "custom"]

LENGTH_PRESETS: dict[str, dict] = {
    "short": {
        "label": "Short",
        "instruction": "Keep every reply brief: one to three short sentences unless the user explicitly asks for more.",
        "max_tokens": 300,
    },
    "normal": {
        "label": "Normal",
        "instruction": "Keep replies concise and to the point, usually under 120 words.",
        "max_tokens": 1024,
    },
    "long": {
        "label": "Long",
        "instruction": "Give complete, detailed answers when the question calls for it.",
        "max_tokens": 4096,
    },
}

VOICE_FRIENDLY_INSTRUCTION = (
    "Your replies are spoken aloud by a text-to-speech voice. Write plain, natural, "
    "conversational sentences: no markdown, headings, bullet lists, tables, emojis or code "
    "blocks, and spell out symbols and abbreviations the way a person would say them."
)


class Persona(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str = "New persona"
    system_prompt: str = "You are a friendly, helpful assistant."
    length: LengthPreset = "short"
    custom_max_tokens: int = Field(default=800, ge=16, le=64000)
    custom_length_instruction: str = ""
    voice_friendly: bool = True
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    reasoning_effort: str | None = "low"
    provider: str = "anthropic"
    model: str = "claude-opus-5-5"
    voice_id: str | None = None
    speak_replies: bool = True
    tts_language: str | None = None  # overrides the voice's language if set


class PersonaOverrides(BaseModel):
    """Per-message tweaks from the chat header that don't modify the persona."""

    length: LengthPreset | None = None
    custom_max_tokens: int | None = None
    temperature: float | None = None
    provider: str | None = None
    model: str | None = None


def compose_system_prompt(p: Persona, length: str | None = None) -> str:
    length = length or p.length
    parts = [p.system_prompt.strip()]
    if length == "custom":
        if p.custom_length_instruction.strip():
            parts.append(p.custom_length_instruction.strip())
    else:
        parts.append(LENGTH_PRESETS[length]["instruction"])
    if p.voice_friendly:
        parts.append(VOICE_FRIENDLY_INSTRUCTION)
    return "\n\n".join(x for x in parts if x)


def max_tokens_for(p: Persona, length: str | None = None, custom: int | None = None) -> int:
    length = length or p.length
    if length == "custom":
        return custom or p.custom_max_tokens
    return LENGTH_PRESETS[length]["max_tokens"]


def build_generation(p: Persona, o: PersonaOverrides | None = None) -> tuple[str, GenerationConfig]:
    """Return (provider_id, GenerationConfig) for a persona plus optional overrides."""
    o = o or PersonaOverrides()
    length = o.length or p.length
    cfg = GenerationConfig(
        model=o.model or p.model,
        system_prompt=compose_system_prompt(p, length),
        max_tokens=max_tokens_for(p, length, o.custom_max_tokens),
        temperature=o.temperature if o.temperature is not None else p.temperature,
        reasoning_effort=p.reasoning_effort or None,
    )
    return (o.provider or p.provider), cfg


DEFAULT_PERSONAS = [
    Persona(
        id="default-short",
        name="Quick & spoken",
        system_prompt="You are a friendly, witty assistant chatting in real time.",
        length="short",
    ),
    Persona(
        id="default-detailed",
        name="Detailed helper",
        system_prompt="You are a knowledgeable, patient assistant.",
        length="long",
        voice_friendly=True,
        reasoning_effort="medium",
    ),
]


class PersonaStore:
    def __init__(self, db: Database):
        self.db = db
        if not self.db.list_personas():
            for p in DEFAULT_PERSONAS:
                self.save(p)
            self.db.set_setting("active_persona_id", DEFAULT_PERSONAS[0].id)

    def list(self) -> list[Persona]:
        return [Persona.model_validate(d) for d in self.db.list_personas()]

    def get(self, pid: str | None) -> Persona | None:
        if not pid:
            return None
        d = self.db.get_persona(pid)
        return Persona.model_validate(d) if d else None

    def save(self, p: Persona) -> Persona:
        self.db.upsert_persona(p.id, p.model_dump())
        return p

    def delete(self, pid: str) -> None:
        self.db.delete_persona(pid)
        if self.db.get_setting("active_persona_id") == pid:
            remaining = self.list()
            self.db.set_setting("active_persona_id", remaining[0].id if remaining else None)

    def duplicate(self, pid: str) -> Persona | None:
        p = self.get(pid)
        if not p:
            return None
        copy = p.model_copy(update={"id": new_id(), "name": f"{p.name} (copy)"})
        return self.save(copy)

    @property
    def active_id(self) -> str | None:
        return self.db.get_setting("active_persona_id")

    def set_active(self, pid: str) -> None:
        self.db.set_setting("active_persona_id", pid)

    def resolve(self, pid: str | None) -> Persona:
        """Persona by id, else the active one, else a built-in default."""
        return self.get(pid) or self.get(self.active_id) or (self.list() or [Persona()])[0]
