HEX_RAINBOW = [
    '#f80c12',
    '#ee1100',
    '#ff3311',
    '#ff4422',
    '#ff6644',
    '#ff9933',
    '#feae2d',
    '#ccbb33',
    '#d0c310',
    '#aacc22',
    '#69d025',
    '#22ccaa',
    '#12bdb9',
    '#11aabb',
    '#4444dd',
    '#3311bb',
    '#3b0cbd',
    '#442299'
]

def get_hex_colors(n_labels):
    """
    Return a list of hex color codes spaced out so for maximum color
    difference.

    args:
        n_labels - number of colors to return
    """
    step = max(1, (len(HEX_RAINBOW) / n_labels))
#    print "COLOR INDS", range(0, len(HEX_RAINBOW), (len(HEX_RAINBOW) / n_labels))
    return HEX_RAINBOW[::step]

