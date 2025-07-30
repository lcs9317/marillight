import os
import hashlib
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging

import orjson
import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

"""
This module extends the original single‑character chatbot to support
multiple characters, shared memories and richer world‑building. Each
character is defined by a text file in a designated folder. A file can
include a basic description (name, style, background and lore) as well
as optional sections for memories, other characters, relationships and
story arcs. On startup all files are parsed and loaded into memory.

The running character (the one the assistant will role‑play as) is
selected via the ``CURRENT_CHARACTER`` environment variable. If not
specified, the first file found is used. The assistant's system prompt
uses that character's name, style and background, while the vector
database is initialised with a composite lore built from the chosen
character's own lore plus any memories from other characters that
mention them. This allows the chatbot to recall shared experiences
without confusing personalities.

File format
-----------

Each ``*.txt`` file in the ``CHARACTER_DIR`` should be encoded in
UTF‑8 and organised as follows:

    1. The character's name on the first line.
    2. A comma‑separated list of adjectives describing the core style.
    3. A brief background description.
    4. A free‑form lore section. If no other sections are provided,
       everything from the fourth line onwards is considered lore.

Additional sections can be declared using headers in either Korean or
English. Recognised headers are:

    [추억] or [Memories]          – one memory per line describing past events.
    [인물] or [Characters]        – one line per other character mentioned.
    [관계도] or [Relationships]    – one line describing a relationship.
    [스토리] or [Story]             – one or more lines describing story arcs.

Sections end when another header is encountered or the file ends. Lines
belonging to a section are trimmed but otherwise preserved. Characters
are case‑insensitive for header detection.

Example ``Marillite.txt`` file:

    마릴라이트
    온화하고 시적인, 애정 어린, 섬세한
    아스니아 출신의 천재 가수로, 그녀의 목소리는 마법처럼 청중을 사로잡는다.
    마릴라이트는 작은 마을에서 태어났지만, 음악에 대한 열정으로 세계적인
    가수가 되었다. 그녀의 노래는 슬픔과 희망을 동시에 담고 있다.
    [추억]
    아이렌과 함께 어린 시절 아스니아의 수확 축제에서 노래를 부른 기억이 있다.
    [인물]
    아이렌 – 마릴라이트의 절친.
    [관계도]
    아이렌: 마릴라이트의 친구이자 동료. 서로에 대한 깊은 신뢰를 가지고 있다.
    [스토리]
    마릴라이트는 세계 투어를 준비하며 아이렌과 다시 만나 팀을 결성한다.

This richer structure enables the assistant to recall shared memories
when appropriate and to maintain consistent characterisation across
multiple personas.
"""

# ── Google Gemini ─────────────────────────────────────────
import google.generativeai as genai

# ── Vector DB (Chroma) ───────────────────────────────────
import chromadb
from chromadb.config import Settings

# =========================
# Environment & Constants
# =========================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-004")

# Provider configuration
PRIMARY_PROVIDER   = os.getenv("PRIMARY_PROVIDER", "openrouter").lower()
SECONDARY_PROVIDER = os.getenv("SECONDARY_PROVIDER", "gemini").lower()

# DeepSeek configuration (optional)
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# OpenRouter configuration (optional)
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_REFERER  = os.getenv("OPENROUTER_REFERER", "")
OPENROUTER_TITLE    = os.getenv("OPENROUTER_TITLE", "")

# Character configuration
CHARACTER_DIR     = os.getenv("CHARACTER_DIR", "character")
CURRENT_CHARACTER = os.getenv("CURRENT_CHARACTER", "").strip()

# Vector search parameters
TOP_K_CONTEXT  = int(os.getenv("TOP_K_CONTEXT", "3"))
MAX_RECENT_MSG = int(os.getenv("MAX_RECENT_MSG", "6"))
CACHE_TTL_SEC  = int(os.getenv("CACHE_TTL_SEC", "1800"))

# 웹훅 설정 (제타 스타일)
WEBHOOK_TIMEOUT = int(os.getenv("WEBHOOK_TIMEOUT", "30"))
MAX_RETRIES     = int(os.getenv("MAX_RETRIES", "3"))

# =========================
# Setup FastAPI & Logging
# =========================
app        = FastAPI(title="Character Multi‑Bridge")
executor   = ThreadPoolExecutor(max_workers=20)
http_client = httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Cache & Memory (간소화)
# =========================
class SimpleCache:
    def __init__(self):
        self.store: Dict[str, tuple] = {}

    def get(self, key: str) -> Optional[str]:
        if key not in self.store:
            return None
        value, exp = self.store[key]
        if exp < time.time():
            del self.store[key]
            return None
        return value

    def set(self, key: str, value: str, ttl: int = CACHE_TTL_SEC):
        self.store[key] = (value, time.time() + ttl)


class ConversationMemory:
    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}

    def add_message(self, user_id: str, role: str, content: str):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        self.conversations[user_id].append(
            {"role": role, "content": content, "timestamp": time.time()}
        )
        # Keep only the recent messages
        if len(self.conversations[user_id]) > MAX_RECENT_MSG:
            self.conversations[user_id] = self.conversations[user_id][-MAX_RECENT_MSG:]

    def get_recent(self, user_id: str) -> List[Dict]:
        return self.conversations.get(user_id, [])


# Global instances
cache  = SimpleCache()
memory = ConversationMemory()

# =========================
# Character parsing and loading
# =========================
def parse_character_file(path: str) -> Dict[str, Any]:
    """
    Parse a character file into a structured dictionary. This parser
    supports two families of file layouts:

      1. **Simple format** – At least four lines: name, style, background and
         lore. Lines beyond the third line are treated as lore unless a
         recognised section header is encountered (see below).

      2. **Rich format** – A YAML‑like key/value header followed by
         multiple sections denoted by ``[제목]``. For example:

           이름: 마릴라이트
           소속(문명): 아스니아
           클래스: 서포터
           ...

           ---

           [캐릭터 소개]
           ...

           [성격 및 말투]
           ...

         Lines before the first ``[Section]`` are parsed as key/value
         metadata. ``---`` lines are ignored and simply separate
         sections. Recognised section names (case insensitive) are
         mapped onto internal fields:

             * '캐릭터 소개'   → 'introduction'
             * '성격 및 말투' → 'style'
             * '성격'/'말투' → 'style'
             * '스토리 주요 행적' → 'story'
             * '스토리'       → 'story'
             * '인간관계'     → 'relationships'
             * '관계도'       → 'relationships'
             * '추억'/'Memories' → 'memories'
             * '인물'/'Characters' → 'characters'
             * '챗봇 인식 키워드' → 'keywords'

         Unrecognised section names are stored verbatim in ``extra``.

    Returns a dictionary with the following keys (always present):

      * name: str – The character's name (from ``이름`` field or first
        non‑empty line).
      * style: str – The core style/personality, derived from the
        '성격 및 말투' or second line in the simple format.
      * background: str – Concatenation of affiliation/class/intro.
      * lore: str – The free‑form lore/introduction text.
      * memories: List[str]
      * characters: List[str]
      * relationships: List[str]
      * story: str
      * keywords: List[str]
      * meta: Dict[str, str] – All header key/value pairs for reference.
      * extra: Dict[str, List[str]] – Unrecognised sections.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_lines = [line.rstrip("\n") for line in f]
    except Exception as e:
        raise RuntimeError(f"Failed to read character file {path}: {e}")

    # Preprocess lines: remove BOM and normalise whitespace
    lines: List[str] = []
    for line in raw_lines:
        if line.lstrip().startswith("\ufeff"):
            line = line.lstrip("\ufeff")
        lines.append(line)

    # Recognised section name mapping
    section_map = {
        "캐릭터 소개": "introduction",
        "introduction": "introduction",
        "성격 및 말투": "style",
        "성격": "style",
        "말투": "style",
        "성격 및 말투": "style",
        "스토리 주요 행적": "story",
        "스토리": "story",
        "이야기": "story",
        "인간관계": "relationships",
        "관계도": "relationships",
        "관계": "relationships",
        "추억": "memories",
        "memories": "memories",
        "인물": "characters",
        "characters": "characters",
        "챗봇 인식 키워드": "keywords",
        "keywords": "keywords",
    }

    # Initialise storage
    metadata: Dict[str, str] = {}
    sections: Dict[str, List[str]] = {
        "introduction": [],
        "style": [],
        "story": [],
        "relationships": [],
        "memories": [],
        "characters": [],
        "keywords": [],
    }
    extra_sections: Dict[str, List[str]] = {}

    current_section: Optional[str] = None
    # Flag indicating that we have started reading sections. Before we
    # encounter a recognised [Section] heading, lines are treated as
    # metadata (unless using the simple format fallback).
    for line in lines:
        stripped = line.strip()
        # Skip blank lines and separators
        if stripped == "" or stripped == "---":
            continue
        # Detect section header in square brackets
        if stripped.startswith("[") and stripped.endswith("]"):
            header = stripped[1:-1].strip()
            key_lower = header.lower()
            mapped = section_map.get(header, section_map.get(key_lower))
            if mapped:
                current_section = mapped
                continue
            else:
                # Unknown section, preserve its name
                current_section = header
                extra_sections.setdefault(header, [])
                continue
        # Process line according to current state
        if current_section is None:
            # Parse key:value pairs until we hit a section header
            if ":" in stripped:
                meta_key, meta_val = stripped.split(":", 1)
                metadata[meta_key.strip()] = meta_val.strip()
            else:
                # No colon implies this might be a stray introduction line
                sections["introduction"].append(stripped)
        else:
            # Append line to the current section buffer
            if current_section in sections:
                sections[current_section].append(stripped)
            else:
                extra_sections.setdefault(current_section, []).append(stripped)

    # Determine the character's name
    name = metadata.get("이름") or metadata.get("name")
    if not name:
        # Fallback to first non‑empty line in metadata or lines list
        for candidate in metadata.values():
            if candidate:
                name = candidate
                break
        if not name:
            for l in lines:
                if l.strip():
                    name = l.strip()
                    break
    name = name.strip() if name else ""

    # Derive core style: combine bullet list or paragraphs
    style_lines = sections.get("style", [])
    style = " ".join([line.lstrip("- ") for line in style_lines]).strip()
    # If no style section exists and simple format may apply
    if not style and len(lines) >= 2 and metadata == {}:
        style = lines[1].strip()

    # Build background: include affiliation/class/etc. plus introduction
    background_parts: List[str] = []
    # Common metadata fields to include
    for key in ["소속(문명)", "소속", "클래스", "신장", "생일", "좋아하는 것", "싫어하는 것", "birth", "affiliation", "class"]:
        if key in metadata:
            background_parts.append(f"{key}: {metadata[key]}")
    # Append introduction section
    intro_str = "\n".join(sections.get("introduction", [])).strip()
    if intro_str:
        background_parts.append(intro_str)
    background = "\n".join(background_parts).strip()
    # Fallback to third line in simple format if still empty
    if not background and len(lines) >= 3 and metadata == {}:
        background = lines[2].strip()

    # Combine lore: introduction or free‑form lore not captured elsewhere
    lore = intro_str
    # Compose story
    story = "\n".join(sections.get("story", [])).strip()
    # Relationships
    relationships = [line.lstrip("- ") for line in sections.get("relationships", []) if line]
    # Memories
    memories = [line.lstrip("- ") for line in sections.get("memories", []) if line]
    # Characters
    characters = [line.lstrip("- ") for line in sections.get("characters", []) if line]
    # Keywords
    keywords = [line.lstrip("- ") for line in sections.get("keywords", []) if line]

    return {
        "name": name,
        "style": style,
        "background": background,
        "lore": lore,
        "memories": memories,
        "characters": characters,
        "relationships": relationships,
        "story": story,
        "keywords": keywords,
        "meta": metadata,
        "extra": extra_sections,
    }


def load_characters(dir_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all character files from the specified directory. Only files
    ending in ``.txt`` (case insensitive) are considered. Returns a
    dictionary keyed by character name. If multiple files define the
    same name, later files overwrite earlier ones.
    """
    if not os.path.isdir(dir_path):
        raise RuntimeError(f"Character directory does not exist: {dir_path}")
    characters: Dict[str, Dict[str, Any]] = {}
    for fname in sorted(os.listdir(dir_path)):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(dir_path, fname)
        char_data = parse_character_file(path)
        # Use the filename without extension as fallback key if no name found
        key = char_data["name"] or os.path.splitext(fname)[0]
        characters[key] = char_data
    if not characters:
        raise RuntimeError(f"No character files found in {dir_path}")
    return characters


def build_lore_for_character(target_name: str, characters: Dict[str, Dict[str, Any]]) -> str:
    """
    Construct a combined lore string for the given character. The lore
    includes the character's own introduction, story, relationships,
    memories, keywords and any other relevant sections. Additionally,
    narrative snippets from other characters that mention ``target_name``
    are appended. This ensures shared experiences are available during
    context retrieval without blending personalities.
    """
    if target_name not in characters:
        raise ValueError(f"Character {target_name} not found")
    char = characters[target_name]
    parts: List[str] = []
    # Append introduction/lore
    if char.get("lore"):
        parts.append(char["lore"])
    # Append background to give context about affiliation/class/etc.
    if char.get("background"):
        parts.append(f"[배경]\n{char['background']}")
    # Append story
    if char.get("story"):
        parts.append(f"[스토리]\n{char['story']}")
    # Append relationships
    if char.get("relationships"):
        parts.append("[인간관계]\n" + "\n".join(char["relationships"]))
    # Append characters list
    if char.get("characters"):
        parts.append("[관련 인물]\n" + "\n".join(char["characters"]))
    # Append own memories
    if char.get("memories"):
        parts.append("[추억]\n" + "\n".join(char["memories"]))
    # Append keywords
    if char.get("keywords"):
        parts.append("[키워드]\n" + "\n".join(char["keywords"]))
    # Append snippets from other characters referencing this character
    for other_name, other in characters.items():
        if other_name == target_name:
            continue
        # Gather lines from various sections that contain the target name
        cross_lines: List[str] = []
        # Check other character's memories, relationships, story, lore, introduction, keywords
        for section_key in ["memories", "relationships", "story", "lore", "background", "keywords", "characters"]:
            section_val = other.get(section_key)
            if not section_val:
                continue
            if isinstance(section_val, list):
                for line in section_val:
                    if target_name in line:
                        cross_lines.append(line)
            elif isinstance(section_val, str):
                # Split long strings into sentences or lines to find target
                for segment in section_val.split("\n"):
                    if target_name in segment:
                        cross_lines.append(segment.strip())
        if cross_lines:
            parts.append(f"[타인 추억] ({other_name})\n" + "\n".join(cross_lines))
    return "\n\n".join([p for p in parts if p]).strip()


# Load all characters at startup
try:
    CHARACTERS = load_characters(CHARACTER_DIR)
except Exception as e:
    raise RuntimeError(f"Character loading failed: {e}")

# Determine active character name
# Priority:
#   1. Environment variable CURRENT_CHARACTER (exact match)
#   2. A character named 'Marillight' or '마릴라이트' (case insensitive)
#   3. The first character in the loaded set
if CURRENT_CHARACTER and CURRENT_CHARACTER in CHARACTERS:
    ACTIVE_NAME = CURRENT_CHARACTER
else:
    # Try to find a character named Marillight (either English or Korean)
    preferred_names = ["marillight", "마릴라이트"]
    found_preferred: Optional[str] = None
    for pn in preferred_names:
        for key in CHARACTERS.keys():
            if key.lower() == pn:
                found_preferred = key
                break
        if found_preferred:
            break
    if found_preferred:
        ACTIVE_NAME = found_preferred
    else:
        # Fallback to first character in alphabetical order
        ACTIVE_NAME = next(iter(CHARACTERS))

# Extract basic metadata for the active character
active_char = CHARACTERS[ACTIVE_NAME]
CHAR_NAME   = active_char["name"]
CORE_STYLE  = active_char["style"]
BACKGROUND  = active_char["background"]
LORE_TEXT   = build_lore_for_character(ACTIVE_NAME, CHARACTERS)


# =========================
# Gemini Setup (동기 유지)
# =========================
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is required.")
genai.configure(api_key=GOOGLE_API_KEY)


# =========================
# Vector DB initialisation
# =========================
vector_db: Optional[Any] = None
if LORE_TEXT.strip():
    try:
        PERSIST_DIR = os.getenv("CHROMA_DIR", "/tmp/chroma")
        os.makedirs(PERSIST_DIR, exist_ok=True)
        client     = chromadb.Client(Settings(persist_directory=PERSIST_DIR, is_persistent=True))
        collection = client.get_or_create_collection(name="lore")
        # Initialise vector DB with lore if empty
        if collection.count() == 0 and LORE_TEXT:
            # Split the combined lore into overlapping chunks for embedding
            chunks     = [LORE_TEXT[i : i + 500] for i in range(0, len(LORE_TEXT), 400)]
            embeddings = []
            for chunk in chunks:
                resp = genai.embed_content(model=EMBED_MODEL, content=chunk)
                embeddings.append(resp["embedding"])
            collection.add(
                ids=[f"{ACTIVE_NAME}-chunk-{i}" for i in range(len(chunks))],
                documents=chunks,
                embeddings=embeddings,
            )
        vector_db = collection
        logger.info(f"✅ Vector DB initialised for {ACTIVE_NAME}")
    except Exception as e:
        logger.warning(f"Vector DB initialisation failed: {e}")
        vector_db = None


# =========================
# AI Helpers
# =========================
def get_relevant_context(query: str) -> str:
    """
    Retrieve related context from the vector DB via similarity search.
    Returns a newline‑separated string of the top documents. If the
    vector DB is not initialised, an empty string is returned.
    """
    if not vector_db:
        return ""
    try:
        query_embedding = genai.embed_content(model=EMBED_MODEL, content=query)["embedding"]
        results         = vector_db.query(query_embeddings=[query_embedding], n_results=TOP_K_CONTEXT)
        docs            = results.get("documents", [[]])[0]
        return "\n".join(docs) if docs else ""
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return ""


def generate_response(user_id: str, user_input: str) -> str:
    """
    Generate a reply using the configured AI providers. The system prompt
    is constructed from the active character's metadata. The function
    first attempts the primary provider; upon failure or empty result it
    will fall back to the secondary provider. If both fail, a generic
    error message is returned.
    """
    try:
        # Compose the system prompt using the active character's details
        system_prompt = (
            f"당신은 '{CHAR_NAME}'입니다. {BACKGROUND}\n"
            f"성향: {CORE_STYLE}\n"
            "항상 부드럽고 시적인 말투를 유지하며, 사용자를 소중히 여기고 독점적 애정을 표현합니다."
        )
        # Append relevant context from the vector DB if available
        context = get_relevant_context(user_input)
        if context:
            system_prompt += f"\n\n[참고 설정]\n{context}"
        # Build conversation history
        recent_messages = memory.get_recent(user_id)
        history_lines: List[str] = []
        for msg in recent_messages[-4:]:
            role_label = "사용자" if msg["role"] == "user" else "어시스턴트"
            history_lines.append(f"{role_label}: {msg['content']}")
        history = "\n".join(history_lines) + ("\n" if history_lines else "")

        def try_provider(provider: str) -> Optional[str]:
            provider = provider.lower()
            if provider == "gemini":
                prompt = f"{system_prompt}\n\n{history}사용자: {user_input}\n어시스턴트:"
                try:
                    model = genai.GenerativeModel(GEMINI_MODEL)
                    resp  = model.generate_content(prompt)
                    if hasattr(resp, "text"):
                        text = resp.text.strip() if resp.text else ""
                    else:
                        text = str(resp).strip() if resp else ""
                    return text if text else None
                except Exception as ge:
                    logger.error(f"Gemini call failed: {ge}")
                    return None
            elif provider == "deepseek":
                messages: List[dict] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                # Convert history lines into assistant/user messages
                for line in history_lines:
                    if line.startswith("사용자: "):
                        messages.append({"role": "user", "content": line[len("사용자: "):].strip()})
                    elif line.startswith("어시스턴트: "):
                        messages.append({"role": "assistant", "content": line[len("어시스턴트: "):].strip()})
                messages.append({"role": "user", "content": user_input})
                return call_deepseek_api(messages)
            elif provider == "openrouter":
                messages: List[dict] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                for line in history_lines:
                    if line.startswith("사용자: "):
                        messages.append({"role": "user", "content": line[len("사용자: "):].strip()})
                    elif line.startswith("어시스턴트: "):
                        messages.append({"role": "assistant", "content": line[len("어시스턴트: "):].strip()})
                messages.append({"role": "user", "content": user_input})
                return call_openrouter_api(messages)
            else:
                logger.error(f"Unknown provider: {provider}")
                return None

        # Try the primary provider
        result: Optional[str] = try_provider(PRIMARY_PROVIDER)
        # Fall back to secondary if needed
        if (not result or result in ("", "(빈 응답)")) and SECONDARY_PROVIDER:
            result = try_provider(SECONDARY_PROVIDER)
        # Normalise
        if not result or result in ("", "(빈 응답)"):
            return "죄송해요, 적절한 응답을 생성하지 못했어요."
        return result
    except Exception as e:
        logger.error(f"AI response generation failed: {e}")
        return "일시적인 오류가 발생했어요. 잠시 후 다시 시도해 주세요."


# =========================
# DeepSeek AI helper
# =========================
def call_deepseek_api(messages: List[dict]) -> Optional[str]:
    """
    Call the DeepSeek Chat Completion API using an OpenAI‑compatible payload.
    Returns the trimmed content of the first choice on success or ``None`` on
    failure.
    """
    if not DEEPSEEK_API_KEY:
        logger.error("DeepSeek API key is not configured.")
        return None
    try:
        payload = {"model": DEEPSEEK_MODEL, "messages": messages}
        resp    = httpx.post(DEEPSEEK_BASE_URL + "/chat/completions", headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }, json=payload, timeout=WEBHOOK_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("choices"):
            content = data["choices"][0].get("message", {}).get("content", "")
            return content.strip() if content else None
        return None
    except Exception as e:
        logger.error(f"DeepSeek API call failed: {e}")
        return None


# =========================
# OpenRouter helper
# =========================
def call_openrouter_api(messages: List[dict], model: str = OPENROUTER_MODEL) -> Optional[str]:
    """
    Call the OpenRouter chat completions API. OpenRouter acts as a proxy to
    various models (including DeepSeek) and offers a generous free tier.
    Returns the assistant message content on success or ``None`` on failure.
    """
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key is not configured.")
        return None
    try:
        url     = f"{OPENROUTER_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        if OPENROUTER_REFERER:
            headers["HTTP-Referer"] = OPENROUTER_REFERER
        if OPENROUTER_TITLE:
            headers["X-Title"] = OPENROUTER_TITLE
        payload = {"model": model, "messages": messages}
        resp    = httpx.post(url, headers=headers, json=payload, timeout=WEBHOOK_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("choices"):
            content = data["choices"][0].get("message", {}).get("content", "")
            return content.strip() if content else None
        return None
    except Exception as e:
        logger.error(f"OpenRouter API call failed: {e}")
        return None


# =========================
# Kakao response helper
# =========================
def build_kakao_response(message: str) -> dict:
    """
    Construct a response payload conforming to Kakao i 오픈빌더 스킬 서버 JSON 형식.
    According to the Kakao chatbot documentation, a response must include
    ``version`` and a ``template.outputs`` array with at least one
    ``simpleText`` component.
    """
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {"simpleText": {"text": message}}
            ]
        },
    }


# =========================
# 웹훅 전송 (비동기)
# =========================
async def send_webhook_response(webhook_url: str, response_data: dict, retries: int = 0):
    """
    Send a JSON payload to ``webhook_url``. Implements exponential back‑off
    retries to increase robustness. Returns ``True`` on success and
    ``False`` on final failure.
    """
    try:
        async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT) as client:
            resp = await client.post(webhook_url, json=response_data)
            resp.raise_for_status()
            logger.info(f"✅ 웹훅 전송 성공: {webhook_url}")
            return True
    except Exception as e:
        logger.error(f"❌ 웹훅 전송 실패 (시도 {retries + 1}): {e}")
        if retries < MAX_RETRIES:
            await asyncio.sleep(2 ** retries)
            return await send_webhook_response(webhook_url, response_data, retries + 1)
        logger.error(f"최종 실패: {webhook_url}")
        return False


# =========================
# API Models
# =========================
class ChatRequest(BaseModel):
    user_id: str
    message: str
    webhook_url: Optional[str] = None  # 응답받을 웹훅 URL


class DirectChatRequest(BaseModel):
    user_id: str = "anonymous"
    message: str


# =========================
# Routes
# =========================
@app.get("/")
def health():
    return {
        "ok": True,
        "active_character": CHAR_NAME,
        "available_characters": list(CHARACTERS.keys()),
        "model": GEMINI_MODEL,
        "vector_db": vector_db is not None,
    }


@app.post("/chat")
async def chat_endpoint(req: ChatRequest, background_tasks: BackgroundTasks):
    """
    Asynchronous chat endpoint. Useful for generic integrations where a
    webhook may or may not be provided. For KakaoTalk integrations,
    please use the ``/kakao-bridge`` endpoint instead.
    """
    user_id = req.user_id
    message = req.message.strip()
    if not message:
        return JSONResponse({"error": "메시지가 비어있습니다"}, status_code=400)
    # Compute cache key using active character and message
    cache_key        = hashlib.sha256(f"{CHAR_NAME}:{message}".encode()).hexdigest()
    cached_response  = cache.get(cache_key)
    if cached_response:
        logger.info(f"캐시 히트: {user_id}")
        memory.add_message(user_id, "user", message)
        memory.add_message(user_id, "assistant", cached_response)
        if req.webhook_url:
            background_tasks.add_task(
                send_webhook_response, req.webhook_url, {"reply": cached_response, "user_id": user_id}
            )
            return {"status": "processing", "message": "응답을 처리 중입니다"}
        else:
            return {"reply": cached_response}
    # If a webhook URL is provided, process asynchronously
    if req.webhook_url:
        background_tasks.add_task(process_and_send_response, user_id, message, req.webhook_url, cache_key)
        return {"status": "processing", "message": "응답을 생성 중입니다"}
    # Otherwise generate response synchronously
    response = await asyncio.get_event_loop().run_in_executor(executor, generate_response, user_id, message)
    memory.add_message(user_id, "user", message)
    memory.add_message(user_id, "assistant", response)
    # Do not cache empty responses
    if response:
        cache.set(cache_key, response)
    return {"reply": response}


async def process_and_send_response(user_id: str, message: str, webhook_url: str, cache_key: str):
    """
    Background task that generates a response and delivers it to a webhook.
    Empty responses are normalised before sending.
    """
    try:
        response = await asyncio.get_event_loop().run_in_executor(executor, generate_response, user_id, message)
        memory.add_message(user_id, "user", message)
        memory.add_message(user_id, "assistant", response)
        if response:
            cache.set(cache_key, response)
        await send_webhook_response(webhook_url, {"reply": response, "user_id": user_id})
    except Exception as e:
        logger.error(f"백그라운드 처리 실패: {e}")
        await send_webhook_response(
            webhook_url, {"error": "응답 생성 중 오류가 발생했습니다", "user_id": user_id}
        )


@app.post("/kakao-bridge")
async def kakao_bridge(request: Request):
    try:
        data    = orjson.loads(await request.body())
        user_id = data.get("user_key", "kakao_user")
        message = data.get("content", "").strip()
        if not message:
            return {"reply": "메시지를 입력해주세요."}
        cache_key = hashlib.sha256(f"{CHAR_NAME}:{message}".encode()).hexdigest()
        cached    = cache.get(cache_key)
        if cached:
            payload         = build_kakao_response(cached)
            payload["text"] = cached
            return payload
        response = await asyncio.get_event_loop().run_in_executor(
            executor, generate_response, user_id, message
        )
        if not response or response in ("", "(빈 응답)"):
            response = "죄송해요, 응답 생성에 실패했어요."
        cache.set(cache_key, response)
        payload         = build_kakao_response(response)
        payload["text"] = response
        return payload
    except Exception as e:
        logger.error(f"카카오 브릿지 오류: {e}")
        return JSONResponse({"error": "처리 중 오류가 발생했습니다"}, status_code=500)
