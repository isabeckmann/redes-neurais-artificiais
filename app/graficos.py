"""Gera um gráfico SVG simples (sem dependências) da curva de MSE por época."""

import math


def salvar_curva_mse_svg(historico, caminho, titulo="MSE por época"):
    largura, altura = 760, 420
    esq, dir_, topo, base = 80, 20, 40, 50
    w, h = largura - esq - dir_, altura - topo - base

    epocas = [r.epoca for r in historico]
    series = [("Treino", "#2563eb", [r.mse_treino for r in historico])]
    if any(r.mse_validacao is not None for r in historico):
        series.append(("Validação", "#dc2626", [r.mse_validacao for r in historico]))

    valores = [v for _, _, vs in series for v in vs if v is not None and v > 0]
    if not valores:
        return
    lmin = math.floor(math.log10(min(valores)))
    lmax = math.ceil(math.log10(max(valores)))
    if lmax == lmin:
        lmax += 1
    emax = max(epocas[-1], 2)

    def px(epoca):
        return esq + (epoca - 1) / (emax - 1) * w

    def py(v):
        return topo + h - (math.log10(v) - lmin) / (lmax - lmin) * h

    partes = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" height="{altura}" '
        f'font-family="sans-serif" font-size="12">',
        f'<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{largura / 2}" y="22" text-anchor="middle" font-size="15">{titulo}</text>',
    ]
    for e in range(lmin, lmax + 1):
        y = py(10 ** e)
        partes.append(f'<line x1="{esq}" y1="{y:.1f}" x2="{esq + w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        partes.append(f'<text x="{esq - 8}" y="{y + 4:.1f}" text-anchor="end">1e{e}</text>')
    for i in range(6):
        ep = 1 + round(i * (emax - 1) / 5)
        x = px(ep)
        partes.append(f'<text x="{x:.1f}" y="{topo + h + 18}" text-anchor="middle">{ep}</text>')
    partes.append(f'<text x="{esq + w / 2}" y="{altura - 8}" text-anchor="middle">Época</text>')
    partes.append(
        f'<text x="18" y="{topo + h / 2}" text-anchor="middle" '
        f'transform="rotate(-90 18 {topo + h / 2})">MSE (escala log)</text>'
    )
    partes.append(f'<rect x="{esq}" y="{topo}" width="{w}" height="{h}" fill="none" stroke="#9ca3af"/>')

    for k, (nome, cor, vs) in enumerate(series):
        pontos = " ".join(
            f"{px(ep):.1f},{py(v):.1f}" for ep, v in zip(epocas, vs) if v is not None and v > 0
        )
        partes.append(f'<polyline points="{pontos}" fill="none" stroke="{cor}" stroke-width="1.6"/>')
        ly = topo + 16 + k * 18
        partes.append(f'<line x1="{esq + w - 110}" y1="{ly - 4}" x2="{esq + w - 90}" y2="{ly - 4}" stroke="{cor}" stroke-width="3"/>')
        partes.append(f'<text x="{esq + w - 84}" y="{ly}">{nome}</text>')
    partes.append("</svg>")

    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(partes))
