import numpy as np
import pytest
import torch

from movie_recs.data import chronological_split
from movie_recs.models import MODELS
from movie_recs.ranking import evaluate, top_k
from movie_recs.train import sample_negatives


def test_split_is_temporal_and_disjoint():
    rows = [
        {"user_id": user, "movie_id": movie, "timestamp": movie // 2}
        for user in (1, 2)
        for movie in reversed(range(20))
    ]
    train, validation, test = chronological_split(rows)
    assert [len(part) for part in (train, validation, test)] == [32, 4, 4]
    for user in (1, 2):
        parts = [
            [r["movie_id"] for r in part if r["user_id"] == user]
            for part in (train, validation, test)
        ]
        assert parts[0] == list(range(16))
        assert parts[1] == [16, 17]
        assert parts[2] == [18, 19]


def test_full_catalog_metrics_match_manual_calculation():
    scores = np.array([[100, 3, 2, 1], [100, 2, 3, 1]], dtype=np.float32)
    metrics = evaluate(scores, [{0}, {0}], [{1, 3}, {1}], k=2)
    discount = 1 / np.log2(3)
    assert metrics["recall@2"] == pytest.approx(0.75)
    assert metrics["ndcg@2"] == pytest.approx((1 / (1 + discount) + discount) / 2)
    assert metrics["evaluated_users"] == 2
    assert top_k(scores[0], range(4), 10).size == 0


def test_negative_sampler_excludes_only_training_observations():
    forbidden = np.array([[True, True, False, False], [False, True, True, False]])
    users = np.repeat([0, 1], 100)
    sampled = sample_negatives(users, forbidden, 4, np.random.default_rng(42))
    assert not forbidden[users[:, None], sampled].any()
    assert set(sampled[:100].flat) == {2, 3}
    assert set(sampled[100:].flat) == {0, 3}
    with pytest.raises(ValueError, match="no unrated"):
        sample_negatives(np.array([0]), np.ones((1, 4), bool), 1, np.random.default_rng(1))


@pytest.mark.parametrize("name", ["mf", "neural"])
def test_exported_towers_preserve_scores(name):
    torch.manual_seed(42)
    model = MODELS[name](2, 4, 8, torch.eye(4))
    with torch.inference_mode():
        user = model.encode_users(torch.tensor([1]))
        items = model.encode_items(torch.arange(4))
        expected = (items * user).sum(dim=-1).numpy()
    np.testing.assert_allclose(items.numpy() @ user.numpy()[0], expected, atol=1e-6)


def test_als_fold_in_matches_official_implicit_user_update():
    from implicit.cpu.als import AlternatingLeastSquares
    from scipy import sparse

    from movie_recs.fold_in import fold_in

    interactions = sparse.csr_matrix(np.array([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=np.float32))
    model = AlternatingLeastSquares(
        factors=3,
        regularization=0.1,
        alpha=20,
        iterations=3,
        random_state=42,
        use_cg=False,
        num_threads=1,
    )
    model.fit(interactions, show_progress=False)
    expected = model.recalculate_user(0, interactions[0])
    actual = fold_in(model.item_factors, [0, 2], regularization=0.1, confidence=20)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
