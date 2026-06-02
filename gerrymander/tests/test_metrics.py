from app.metrics import efficiency_gap, mean_median, seats_won, polsby_popper


def test_seats_won_basic():
    votes = {0: (60, 40), 1: (30, 70), 2: (55, 45)}
    assert seats_won(votes) == (2, 1)


def test_efficiency_gap_neutral_when_symmetric():
    # Two perfectly mirrored districts → zero EG.
    votes = {0: (60, 40), 1: (40, 60)}
    assert abs(efficiency_gap(votes)) < 1e-9


def test_efficiency_gap_sign_pro_dem():
    # D wins narrowly twice, R wins by a landslide once → R has wasted votes, EG > 0 (pro-D).
    votes = {0: (51, 49), 1: (51, 49), 2: (10, 90)}
    assert efficiency_gap(votes) > 0


def test_mean_median_pro_rep_when_d_share_skewed_high():
    # D share distribution skewed by one packed district → mean > median → positive (pro-D).
    votes = {0: (90, 10), 1: (40, 60), 2: (40, 60)}
    assert mean_median(votes) > 0


def test_polsby_popper_circle_is_one():
    import math
    r = 1.0
    area = math.pi * r * r
    perim = 2 * math.pi * r
    assert abs(polsby_popper(area, perim) - 1.0) < 1e-9


def test_polsby_popper_square_below_one():
    # Square: area = 1, perimeter = 4 → 4π/16 ≈ 0.785
    assert abs(polsby_popper(1.0, 4.0) - 0.7853981633974483) < 1e-9
