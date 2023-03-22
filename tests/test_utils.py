from ml_swissknife import utils


def test_unnest_pytree():
    l = 1
    unnested_l, nest = utils.unnest_pytree(l)
    l_recover = nest(unnested_l)
    assert l == l_recover

    l = [1, 2, 3]
    unnested_l, nest = utils.unnest_pytree(l)
    l_recover = nest(unnested_l)
    assert unnested_l == [1, 2, 3]
    assert l == l_recover

    l = [[1, 2, [3, [4, 5]]], 6, [7, 8]]
    unnested_l, nest = utils.unnest_pytree(l)
    l_recover = nest(unnested_l)
    assert unnested_l == [1, 2, 3, 4, 5, 6, 7, 8]
    assert l == l_recover
