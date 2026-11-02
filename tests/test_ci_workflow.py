"""What the test workflow has to keep true for the tokenizer cache to work (#200).

Read as text rather than parsed: PyYAML is in nobody's requirements.txt, only
in the environment transitively, so a test that imported it would pass here and
be the one thing missing on a fresh install.
"""

import re
from pathlib import Path

from src.retrieval.constants import EMBED_MODEL

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
CONSTANTS = (ROOT / "src" / "retrieval" / "constants.py").read_text(encoding="utf-8")

# The sed expression the workflow names the cache with, as its Python twin:
#   sed -n 's/^EMBED_MODEL = "\\(.*\\)"$/\\1/p'
AS_SED_READS_IT = re.compile(r'^EMBED_MODEL = "(.*)"$', re.MULTILINE)

AN_ACTION = re.compile(r"^\s*(?:- )?uses:\s*(\S+)", re.MULTILINE)

# The workflow with its comments taken out, for asking what it does rather than
# what it says about itself: the comment on the cache step names restore-keys
# in order to say it is not used.
SETTINGS = "\n".join(line for line in WORKFLOW.splitlines()
                     if not line.lstrip().startswith("#"))


def test_the_cache_key_reads_the_model_python_uses():
    """The key comes from sed over constants.py, so the two have to agree.

    Written another way -- a type annotation, single quotes, a second
    assignment -- sed returns nothing or something else, and the job stops at
    ``test -n``. That is the right failure, but it happens only once a run
    reaches CI. This fails on the laptop that wrote it.
    """
    assert AS_SED_READS_IT.findall(CONSTANTS) == [EMBED_MODEL]


def test_the_cache_key_carries_the_model_and_nothing_restores_past_it():
    """A change of embedding model has to miss the old entry.

    ``restore-keys`` is the way that quietly stops being true: a prefix key
    matches the previous model's tokenizer, which is the one entry this cache
    must never hand back.
    """
    assert ("key: huggingface-${{ runner.os }}-"
            "${{ steps.tokenizer.outputs.embed-model }}") in SETTINGS
    assert "restore-keys" not in SETTINGS


def test_every_action_is_pinned_to_a_commit_sha():
    """As #145 did for the others: a tag can be moved to other code, a commit
    cannot."""
    actions = AN_ACTION.findall(WORKFLOW)
    assert actions, "no actions found, so this test is checking nothing"
    for action in actions:
        _, _, pinned_to = action.partition("@")
        assert re.fullmatch(r"[0-9a-f]{40}", pinned_to), action
