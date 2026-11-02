"""Author-selected teal/coral palette; opaque print-safe magnitude tints."""
TEAL = '#00B887'
CORAL = '#FF5B61'
BLUE = '#387ED1'
PURPLE = '#8066BA'
GOLD = '#D89A33'
GREY = '#E7EBEF'
PALETTE = ('00B887', '33C69F', '66D4B7', '99E3CF', 'BFEDE1', 'E5F8F3',
           'FFF0EF', 'FFD6D8', 'FFBDC0', 'FF9DA2', 'FF7C81', 'FF5B61')


def tint(color, strength):
    values = [int(color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)]
    return tuple(1-strength*(1-value) for value in values)
