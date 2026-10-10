"""Entry point: python submit.py [--space NAME] [--yes]  (same as python -m gaia_agent.submit).

Dry run unless --yes is passed.
"""

from gaia_agent.submit import main

if __name__ == "__main__":
    main()
