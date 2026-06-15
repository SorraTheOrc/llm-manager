import re
from pathlib import Path
import math


def hex_to_rgb(hex_str: str):
    s = hex_str.strip()
    if s.startswith('#'):
        s = s[1:]
    if len(s) == 3:
        s = ''.join([c*2 for c in s])
    if len(s) != 6:
        raise ValueError(f"Unsupported hex color: {hex_str}")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b)


def srgb_channel_to_linear(c: float) -> float:
    c = c / 255.0
    if c <= 0.03928:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def luminance(rgb):
    r, g, b = rgb
    R = srgb_channel_to_linear(r)
    G = srgb_channel_to_linear(g)
    B = srgb_channel_to_linear(b)
    return 0.2126 * R + 0.7152 * G + 0.0722 * B


def contrast_ratio(rgb1, rgb2):
    L1 = luminance(rgb1)
    L2 = luminance(rgb2)
    L_high = max(L1, L2)
    L_low = min(L1, L2)
    return (L_high + 0.05) / (L_low + 0.05)


def resolve_var(vars_map, val):
    """Resolve a CSS value which may be a var(--name, fallback) expression or a hex/rgb value."""
    val = val.strip()
    # handle var(--name, fallback) or var(--name)
    m = re.match(r"var\(\s*--([a-z0-9\-]+)\s*(?:,\s*([^\)]+)\s*)?\)", val)
    if m:
        name = m.group(1)
        fallback = m.group(2)
        if name in vars_map:
            return resolve_var(vars_map, vars_map[name])
        if fallback:
            return resolve_var(vars_map, fallback)
        raise ValueError(f"Unresolved var: --{name}")

    # rgba(...) -> ignore alpha and return rgb tuple
    m2 = re.match(r"rgba?\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)(?:\s*,\s*([0-9\.]+))?\s*\)", val)
    if m2:
        r = int(m2.group(1)); g = int(m2.group(2)); b = int(m2.group(3))
        return (r, g, b)

    # hex color
    if val.startswith('#'):
        return hex_to_rgb(val)

    # direct hex without hash
    if re.match(r'^[0-9a-fA-F]{6}$', val):
        return hex_to_rgb('#' + val)

    # fall back: try to strip quotes
    v2 = val.strip('"\'')
    if v2.startswith('#'):
        return hex_to_rgb(v2)

    raise ValueError(f"Unsupported color format: {val}")


def extract_css_vars_from_server() -> dict:
    vars_map = {}
    # base is .../proxy (one level up from proxy/tests/)
    base = Path(__file__).resolve().parents[1]
    # Source files live under proxy/proxy/
    for pyfile in ['server.py', 'ui.py']:
        p = base / 'proxy' / pyfile
        if not p.exists():
            continue
        txt = p.read_text(encoding='utf-8')
        for m in re.finditer(r'--([a-z0-9\-]+):\s*([^;]+);', txt, flags=re.IGNORECASE):
            name = m.group(1)
            val = m.group(2).strip()
            vars_map[name] = val
    # Template files live under proxy/templates/
    templates_dir = base / 'templates'
    if templates_dir.exists():
        for tf in templates_dir.iterdir():
            if tf.suffix in ('.html', '.htm'):
                txt = tf.read_text(encoding='utf-8')
                for m in re.finditer(r'--([a-z0-9\-]+):\s*([^;]+);', txt, flags=re.IGNORECASE):
                    name = m.group(1)
                    val = m.group(2).strip()
                    vars_map[name] = val
    return vars_map


def test_log_badge_contrast():
    vars_map = extract_css_vars_from_server()

    # resolve badge text color (default)
    try:
        badge_text = resolve_var(vars_map, vars_map.get('log-badge-color', 'var(--text-primary, #eee)'))
    except Exception:
        # fallback to text-primary
        badge_text = resolve_var(vars_map, vars_map.get('text-primary', '#eee'))

    # Ensure badge_text is rgb tuple
    if isinstance(badge_text, tuple):
        badge_text_rgb = badge_text
    else:
        badge_text_rgb = resolve_var(vars_map, badge_text)

    # Severities to check
    severities = {
        'error': 'log-error',
        'warning': 'log-warning',
        'info': 'log-info',
        'debug': 'log-debug'
    }

    for sev, varname in severities.items():
        assert varname in vars_map, f"Missing CSS variable: --{varname}"
        bg = resolve_var(vars_map, vars_map[varname])
        if isinstance(bg, tuple):
            bg_rgb = bg
        else:
            bg_rgb = resolve_var(vars_map, bg)

        # Map to the runtime badge text colors implemented in the client-side JS
        # (error: white, others: black)
        if sev == 'error':
            text_rgb = hex_to_rgb('#ffffff')
        else:
            text_rgb = hex_to_rgb('#000000')

        cr = contrast_ratio(text_rgb, bg_rgb)
        # Expect at least 4.5:1 contrast for badge text vs badge background
        assert cr >= 4.5, f"Contrast too low for {sev} badge: {cr:.2f} against bg {bg_rgb}"
