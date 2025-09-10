"""Dummy test."""

from folktexts import __version__


def test_version_in_alpha():
    """Checks whether we're still in alpha."""
    assert __version__.startswith("0.")
