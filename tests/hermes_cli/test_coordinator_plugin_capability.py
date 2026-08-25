from hermes_cli.plugin_capabilities import (
    VALID_CAPABILITY_IDS,
    parse_declared_capabilities,
    plugin_capability_granted,
)


def test_coordinator_capability_is_known_but_has_no_legacy_bypass():
    capability = "delegation.coordinator"
    assert capability in VALID_CAPABILITY_IDS
    assert parse_declared_capabilities([capability], "kospi") == [capability]
    assert not plugin_capability_granted(
        "kospi-team",
        capability,
        config={
            "plugins": {
                "entries": {
                    "kospi-team": {
                        "allow_coordinator": True,
                    }
                }
            }
        },
    )


def test_coordinator_capability_requires_explicit_grant():
    capability = "delegation.coordinator"
    assert plugin_capability_granted(
        "kospi-team",
        capability,
        config={
            "plugins": {
                "entries": {
                    "kospi-team": {
                        "granted_capabilities": [capability],
                    }
                }
            }
        },
    )


def test_generic_subagent_capability_is_separate_and_grant_only():
    capability = "delegation.subagents"
    assert capability in VALID_CAPABILITY_IDS
    assert parse_declared_capabilities([capability], "generic") == [capability]
    assert not plugin_capability_granted(
        "generic",
        capability,
        config={"plugins": {"entries": {"generic": {"allow_subagents": True}}}},
    )
    assert plugin_capability_granted(
        "generic",
        capability,
        config={
            "plugins": {
                "entries": {
                    "generic": {"granted_capabilities": [capability]},
                }
            }
        },
    )
