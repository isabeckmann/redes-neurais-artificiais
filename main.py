"""Ponto de entrada da aplicação.

    python main.py            → menu interativo
    python main.py --help     → ajuda da linha de comando
"""

import sys

from app.cli import main

if __name__ == "__main__":
    # Evita UnicodeEncodeError ao redirecionar a saída no Windows (acentos e setas).
    for fluxo in (sys.stdout, sys.stderr):
        reconfigurar = getattr(fluxo, "reconfigure", None)
        if callable(reconfigurar):
            reconfigurar(encoding="utf-8", errors="replace")
    sys.exit(main())