"""Author-selected teal/coral palette; opaque print-safe magnitude tints."""
TEAL = '#00B887'
CORAL = '#FF5B61'
BLUE = '#387ED1'
PURPLE = '#8066BA'
GOLD = '#D89A33'
GREY = '#E7EBEF'
# Keep every tier saturated: avoid the former near-white pink/green appearance.
PALETTE = ('00B887', '0DBC8D', '1ABF93', '26C399', '33C69F', '47CCA9', 'FF898D', 'FF7C81', 'FF7479', 'FF6B71', 'FF6369', 'FF5B61')


def tint(color, strength):
    values = [int(color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)]
    return tuple(1-strength*(1-value) for value in values)
