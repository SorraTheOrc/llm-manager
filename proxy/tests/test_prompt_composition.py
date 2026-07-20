"""
Prompt composition tests.

Unit tests that validate message composition logic:
- `override` replaces system messages with file content.
- `prepend` inserts file content as the first system message.
"""

import pytest

# ---------------------------------------------------------------------------
# Test: compose_messages function exists
# ---------------------------------------------------------------------------

def test_compose_messages_module_importable():
    """The compose_messages function should be importable."""
    from proxy.prompt_resolver import compose_messages
    assert callable(compose_messages)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_messages():
    """Standard set of messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Tell me a joke."},
    ]


@pytest.fixture
def no_system_messages():
    """Messages without a system prompt."""
    return [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi!"},
    ]


@pytest.fixture
def multiple_system_messages():
    """Multiple system messages (some clients send more than one)."""
    return [
        {"role": "system", "content": "First system."},
        {"role": "system", "content": "Second system."},
        {"role": "user", "content": "Hello!"},
    ]


# ---------------------------------------------------------------------------
# Test: Override mode replaces all system messages
# ---------------------------------------------------------------------------

def test_override_replaces_system_messages(sample_messages):
    """In override mode, all system messages should be replaced."""
    from proxy.prompt_resolver import compose_messages

    prompt_result = {
        "content": "You are a coding assistant.",
        "mode": "override",
        "source": "/path/to/prompt.txt",
    }

    result = compose_messages(sample_messages, prompt_result)

    # Should have one system message with the new content
    system_msgs = [m for m in result if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "You are a coding assistant."

    # Non-system messages should be preserved
    non_system = [m for m in result if m["role"] != "system"]
    assert len(non_system) == 3
    assert non_system[0]["content"] == "Hello!"
    assert non_system[1]["content"] == "Hi there!"
    assert non_system[2]["content"] == "Tell me a joke."


def test_override_with_multiple_system_messages(multiple_system_messages):
    """In override mode, all system messages should be replaced."""
    from proxy.prompt_resolver import compose_messages

    prompt_result = {
        "content": "You are an agent.",
        "mode": "override",
        "source": "/path/to/agent.txt",
    }

    result = compose_messages(multiple_system_messages, prompt_result)

    system_msgs = [m for m in result if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "You are an agent."

    # User messages preserved
    user_msgs = [m for m in result if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "Hello!"


def test_override_with_no_system_messages(no_system_messages):
    """In override mode with no existing system messages, one should be added."""
    from proxy.prompt_resolver import compose_messages

    prompt_result = {
        "content": "You are an assistant.",
        "mode": "override",
        "source": "/path/to/assistant.txt",
    }

    result = compose_messages(no_system_messages, prompt_result)

    system_msgs = [m for m in result if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "You are an assistant."

    # System message should be first
    assert result[0]["role"] == "system"


# ---------------------------------------------------------------------------
# Test: Prepend mode inserts prompt before existing system messages
# ---------------------------------------------------------------------------

def test_prepend_inserts_before_system(sample_messages):
    """In prepend mode, prompt content should be the first system message."""
    from proxy.prompt_resolver import compose_messages

    prompt_result = {
        "content": "IMPORTANT: Be concise.",
        "mode": "prepend",
        "source": "/path/to/important.txt",
    }

    result = compose_messages(sample_messages, prompt_result)

    # First message should be the prepended prompt
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "IMPORTANT: Be concise."

    # Original system message should still be present
    system_msgs = [m for m in result if m["role"] == "system"]
    assert len(system_msgs) == 2
    assert system_msgs[1]["content"] == "You are a helpful assistant."

    # Non-system messages unchanged
    non_system = [m for m in result if m["role"] != "system"]
    assert len(non_system) == 3


def test_prepend_with_multiple_system_messages(multiple_system_messages):
    """Prepend should insert before all existing system messages."""
    from proxy.prompt_resolver import compose_messages

    prompt_result = {
        "content": "PREFIX: Instruction.",
        "mode": "prepend",
        "source": "/path/to/prefix.txt",
    }

    result = compose_messages(multiple_system_messages, prompt_result)

    # System messages: prepended prompt, then original ones
    system_msgs = [m for m in result if m["role"] == "system"]
    assert len(system_msgs) == 3
    assert system_msgs[0]["content"] == "PREFIX: Instruction."
    assert system_msgs[1]["content"] == "First system."
    assert system_msgs[2]["content"] == "Second system."


def test_prepend_with_no_system_messages(no_system_messages):
    """Prepend with no existing system messages should add one at the start."""
    from proxy.prompt_resolver import compose_messages

    prompt_result = {
        "content": "You are a helpful robot.",
        "mode": "prepend",
        "source": "/path/to/robot.txt",
    }

    result = compose_messages(no_system_messages, prompt_result)

    system_msgs = [m for m in result if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "You are a helpful robot."
    assert result[0]["role"] == "system"


# ---------------------------------------------------------------------------
# Test: When prompt_result is None, messages pass through unchanged
# ---------------------------------------------------------------------------

def test_no_prompt_returns_unchanged(sample_messages):
    """When no prompt is resolved, messages should pass through unchanged."""
    from proxy.prompt_resolver import compose_messages

    result = compose_messages(sample_messages, None)
    assert result == sample_messages


# ---------------------------------------------------------------------------
# Test: Empty messages handling
# ---------------------------------------------------------------------------

def test_empty_messages_with_override():
    """Override with empty messages should add a system message."""
    from proxy.prompt_resolver import compose_messages

    prompt_result = {
        "content": "You are empty slate.",
        "mode": "override",
        "source": "/path/to/prompt.txt",
    }

    result = compose_messages([], prompt_result)
    assert len(result) == 1
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are empty slate."


def test_empty_messages_with_prepend():
    """Prepend with empty messages should add a system message."""
    from proxy.prompt_resolver import compose_messages

    prompt_result = {
        "content": "You are empty slate.",
        "mode": "prepend",
        "source": "/path/to/prompt.txt",
    }

    result = compose_messages([], prompt_result)
    assert len(result) == 1
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are empty slate."


# ---------------------------------------------------------------------------
# Test: Preserves message ordering
# ---------------------------------------------------------------------------

def test_preserves_message_ordering(sample_messages):
    """Non-system messages should retain their relative order."""
    from proxy.prompt_resolver import compose_messages

    prompt_result = {
        "content": "New system.",
        "mode": "override",
        "source": "/path/to/new.txt",
    }

    result = compose_messages(sample_messages, prompt_result)

    non_system = [m for m in result if m["role"] != "system"]
    assert non_system[0] == {"role": "user", "content": "Hello!"}
    assert non_system[1] == {"role": "assistant", "content": "Hi there!"}
    assert non_system[2] == {"role": "user", "content": "Tell me a joke."}


# ---------------------------------------------------------------------------
# Test: Override does not duplicate system messages
# ---------------------------------------------------------------------------

def test_override_does_not_duplicate_system(sample_messages):
    """Override should replace, not duplicate, system messages."""
    from proxy.prompt_resolver import compose_messages

    prompt_result = {
        "content": "Single system message.",
        "mode": "override",
        "source": "/path/to/single.txt",
    }

    result = compose_messages(sample_messages, prompt_result)
    system_msgs = [m for m in result if m["role"] == "system"]
    assert len(system_msgs) == 1
