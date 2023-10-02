import numpy as np
import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


def seismic_labels(x_coord, y_coord, z_coord, subcube_dim, label_cube_dim):
    return np.array(
        data.seismic_labels(
            trace="abc",
            x_coord=x_coord,
            y_coord=y_coord,
            z_coord=z_coord,
            subcube_dim=subcube_dim,
            label_cube_dim=label_cube_dim,
            max_label=100_000,
        )
    ).reshape([subcube_dim // label_cube_dim] * 3)


def test_seismic_label_hash_consistency():
    dim = 20
    label_cube_dim = 5

    expected_labels = seismic_labels(
        x_coord=0,
        y_coord=0,
        z_coord=0,
        subcube_dim=dim,
        label_cube_dim=label_cube_dim,
    )

    for i in range(0, dim, label_cube_dim):
        for j in range(0, dim, label_cube_dim):
            for k in range(0, dim, label_cube_dim):
                label = seismic_labels(
                    x_coord=i,
                    y_coord=j,
                    z_coord=k,
                    subcube_dim=label_cube_dim,
                    label_cube_dim=label_cube_dim,
                )[0, 0, 0]

                expected = expected_labels[
                    i // label_cube_dim, j // label_cube_dim, k // label_cube_dim
                ]
                assert expected == label


def test_seismic_label_hash_overlap():
    X, Y, Z = 40, 50, 60

    label_cube_dim = 5
    subcube_dim = 20

    expected_labels = seismic_labels(
        x_coord=0,
        y_coord=0,
        z_coord=0,
        subcube_dim=max(X, Y, Z),
        label_cube_dim=label_cube_dim,
    )

    for i in range(0, X, label_cube_dim):
        for j in range(0, Y, label_cube_dim):
            for k in range(0, Z, label_cube_dim):
                if (
                    (i + subcube_dim > X)
                    or (j + subcube_dim > Y)
                    or (k + subcube_dim) > Z
                ):
                    continue

                labels = seismic_labels(
                    x_coord=i,
                    y_coord=j,
                    z_coord=k,
                    subcube_dim=subcube_dim,
                    label_cube_dim=label_cube_dim,
                )

                label_i = i // label_cube_dim
                label_j = j // label_cube_dim
                label_k = k // label_cube_dim
                labels_per_side = subcube_dim // label_cube_dim

                expected = expected_labels[
                    label_i : label_i + labels_per_side,
                    label_j : label_j + labels_per_side,
                    label_k : label_k + labels_per_side,
                ]

                assert np.array_equal(labels, expected)


def test_seismic_spatial_overlap():
    hashes_1 = seismic_labels(
        x_coord=10, y_coord=20, z_coord=30, subcube_dim=20, label_cube_dim=4
    )

    hashes_2 = seismic_labels(
        x_coord=10, y_coord=24, z_coord=30, subcube_dim=20, label_cube_dim=4
    )

    assert np.array_equal(hashes_1[:, 1:, :], hashes_2[:, :-1, :])

    hashes_3 = seismic_labels(
        x_coord=18, y_coord=24, z_coord=30, subcube_dim=20, label_cube_dim=4
    )

    assert np.array_equal(hashes_1[2:, 1:, :], hashes_3[:-2, :-1, :])

    hashes_3 = seismic_labels(
        x_coord=14, y_coord=32, z_coord=26, subcube_dim=20, label_cube_dim=4
    )

    assert np.array_equal(hashes_2[1:, 2:, :-1], hashes_3[:-1, :-2, 1:])
