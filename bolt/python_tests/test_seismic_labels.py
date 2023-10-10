import numpy as np
import pytest
from thirdai import bolt

pytestmark = [pytest.mark.unit]


def seismic_labels(x_coord, y_coord, z_coord, subcube_shape, label_cube_shape):
    return np.array(
        bolt.seismic.seismic_labels(
            volume="abc",
            x_coord=x_coord,
            y_coord=y_coord,
            z_coord=z_coord,
            subcube_shape=subcube_shape,
            label_cube_shape=label_cube_shape,
            max_label=100_000,
        )
    ).reshape([subcube_shape // label_cube_shape] * 3)


def test_seismic_label_hash_consistency():
    dim = 20
    label_cube_shape = 5

    expected_labels = seismic_labels(
        x_coord=0,
        y_coord=0,
        z_coord=0,
        subcube_shape=dim,
        label_cube_shape=label_cube_shape,
    )

    for i in range(0, dim, label_cube_shape):
        for j in range(0, dim, label_cube_shape):
            for k in range(0, dim, label_cube_shape):
                label = seismic_labels(
                    x_coord=i,
                    y_coord=j,
                    z_coord=k,
                    subcube_shape=label_cube_shape,
                    label_cube_shape=label_cube_shape,
                )[0, 0, 0]

                expected = expected_labels[
                    i // label_cube_shape, j // label_cube_shape, k // label_cube_shape
                ]
                assert expected == label


def test_seismic_label_hash_overlap():
    X, Y, Z = 40, 50, 60

    label_cube_shape = 5
    subcube_shape = 20

    expected_labels = seismic_labels(
        x_coord=0,
        y_coord=0,
        z_coord=0,
        subcube_shape=max(X, Y, Z),
        label_cube_shape=label_cube_shape,
    )

    for i in range(0, X, label_cube_shape):
        for j in range(0, Y, label_cube_shape):
            for k in range(0, Z, label_cube_shape):
                if (
                    (i + subcube_shape > X)
                    or (j + subcube_shape > Y)
                    or (k + subcube_shape) > Z
                ):
                    continue

                labels = seismic_labels(
                    x_coord=i,
                    y_coord=j,
                    z_coord=k,
                    subcube_shape=subcube_shape,
                    label_cube_shape=label_cube_shape,
                )

                label_i = i // label_cube_shape
                label_j = j // label_cube_shape
                label_k = k // label_cube_shape
                labels_per_side = subcube_shape // label_cube_shape

                expected = expected_labels[
                    label_i : label_i + labels_per_side,
                    label_j : label_j + labels_per_side,
                    label_k : label_k + labels_per_side,
                ]

                assert np.array_equal(labels, expected)


def test_seismic_spatial_overlap():
    hashes_1 = seismic_labels(
        x_coord=10, y_coord=20, z_coord=30, subcube_shape=20, label_cube_shape=4
    )

    hashes_2 = seismic_labels(
        x_coord=10, y_coord=24, z_coord=30, subcube_shape=20, label_cube_shape=4
    )

    assert np.array_equal(hashes_1[:, 1:, :], hashes_2[:, :-1, :])

    hashes_3 = seismic_labels(
        x_coord=18, y_coord=24, z_coord=30, subcube_shape=20, label_cube_shape=4
    )

    assert np.array_equal(hashes_1[2:, 1:, :], hashes_3[:-2, :-1, :])

    hashes_3 = seismic_labels(
        x_coord=14, y_coord=32, z_coord=26, subcube_shape=20, label_cube_shape=4
    )

    assert np.array_equal(hashes_2[1:, 2:, :-1], hashes_3[:-1, :-2, 1:])
