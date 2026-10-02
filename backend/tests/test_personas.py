from vctts.personas import (
    LENGTH_PRESETS,
    VOICE_FRIENDLY_INSTRUCTION,
    Persona,
    PersonaOverrides,
    PersonaStore,
    build_generation,
)
from vctts.storage import Database


def test_short_persona_prompt_and_tokens():
    p = Persona(system_prompt="You are Bob.", length="short", provider="openai", model="m1", temperature=0.4)
    provider, cfg = build_generation(p)
    assert provider == "openai"
    assert cfg.model == "m1"
    assert cfg.system_prompt.startswith("You are Bob.")
    assert LENGTH_PRESETS["short"]["instruction"] in cfg.system_prompt
    assert VOICE_FRIENDLY_INSTRUCTION in cfg.system_prompt
    assert cfg.max_tokens == LENGTH_PRESETS["short"]["max_tokens"]
    assert cfg.temperature == 0.4


def test_custom_length_and_no_voice_friendly():
    p = Persona(length="custom", custom_max_tokens=123, custom_length_instruction="Answer in haiku.",
                voice_friendly=False)
    _, cfg = build_generation(p)
    assert cfg.max_tokens == 123
    assert "haiku" in cfg.system_prompt
    assert VOICE_FRIENDLY_INSTRUCTION not in cfg.system_prompt


def test_overrides_win():
    p = Persona(length="short", provider="anthropic", model="a")
    provider, cfg = build_generation(p, PersonaOverrides(length="long", provider="gemini", model="g"))
    assert provider == "gemini" and cfg.model == "g"
    assert cfg.max_tokens == LENGTH_PRESETS["long"]["max_tokens"]
    assert LENGTH_PRESETS["long"]["instruction"] in cfg.system_prompt


def test_store_defaults_and_crud(tmp_path):
    store = PersonaStore(Database(tmp_path / "db.sqlite"))
    ps = store.list()
    assert len(ps) == 2 and store.active_id == ps[0].id
    copy = store.duplicate(ps[0].id)
    assert copy.name.endswith("(copy)")
    copy.system_prompt = "Changed"
    store.save(copy)
    assert store.get(copy.id).system_prompt == "Changed"
    store.delete(ps[0].id)
    assert store.active_id != ps[0].id
    assert store.resolve(None).id == store.active_id
