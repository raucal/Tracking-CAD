"""Scene-specific detection policy, independent of manifest ground truth."""
VERTICAL_OPTION_A = '03a_vertical_option_a'
CORNER_X_MIN = 0.72
CORNER_Y_MIN = 0.65
CORNER_FILTER_REASON = 'vertical_option_a_lower_right_corner'


def reject_vertical_option_a_corner(part_key, bbox, image_size):
    """Reject only CAD 03a with box centre in the lower-right scene region.

    bbox is pixel xywh; image_size is (width, height). Comparisons are strict.
    Objects elsewhere and other CAD identities remain eligible.
    """
    if part_key != VERTICAL_OPTION_A or bbox is None or len(bbox) != 4:
        return False
    x, y, width, height = map(float, bbox)
    image_w, image_h = image_size
    if image_w <= 0 or image_h <= 0 or width <= 0 or height <= 0:
        return False
    return (x + width / 2) / image_w > CORNER_X_MIN and (y + height / 2) / image_h > CORNER_Y_MIN


def rank_candidates(rows, bbox, image_size, enabled=True):
    """Preserve measured scores/raw ranks while ranking eligible candidates first."""
    rows.sort(key=lambda r: (-r['score'], r['category_id']))
    for raw_rank, row in enumerate(rows, 1):
        excluded = enabled and reject_vertical_option_a_corner(row['cad_key'], bbox, image_size)
        row.update(raw_rank=raw_rank, eligible=not excluded,
                   filter_reason=CORNER_FILTER_REASON if excluded else None)
    eligible_semantic = sorted((r for r in rows if r['eligible']), key=lambda r: (-r['semantic'], r['category_id']))
    semantic_ranks = {r['cad_key']:i+1 for i,r in enumerate(eligible_semantic)}
    rows.sort(key=lambda r: (not r['eligible'], -r['score'], r['category_id']))
    for rank, row in enumerate(rows, 1):
        row.update(rank=rank if row['eligible'] else None, semantic_rank=semantic_ranks.get(row['cad_key']))
    return rows
