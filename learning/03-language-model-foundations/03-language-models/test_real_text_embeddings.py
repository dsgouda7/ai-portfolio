"""Offline contracts for the shared windows and their objective-specific pairs."""

import unittest

import numpy as np
import torch

from real_text_embeddings import make_document_windows, objective_pairs


class DocumentWindowTests(unittest.TestCase):
    def test_sliding_preserves_order_repeats_and_unknown_slots(self):
        actual = make_document_windows([[1, 0, 1, 3, 4, 5, 6]], 5)
        np.testing.assert_array_equal(
            actual, [[1, 0, 1, 3, 4], [0, 1, 3, 4, 5], [1, 3, 4, 5, 6]],
        )
        self.assertEqual(actual.dtype, np.int64)

    def test_windows_never_cross_document_boundaries(self):
        actual = make_document_windows([[1, 2, 3], [], [9], [4, 5, 6, 7]], 3)
        np.testing.assert_array_equal(actual, [[1, 2, 3], [4, 5, 6], [5, 6, 7]])

    def test_empty_and_short_documents_keep_two_axes(self):
        for documents in ([], [[]], [[1, 2], [3, 4]]):
            with self.subTest(documents=documents):
                actual = make_document_windows(documents, 5)
                self.assertEqual(actual.shape, (0, 5))
                self.assertEqual(actual.dtype, np.int64)

    def test_width_one_keeps_every_token(self):
        np.testing.assert_array_equal(
            make_document_windows([[3, 0], [7]], 1), [[3], [0], [7]],
        )

    def test_nonpositive_width_is_an_error(self):
        for width in (0, -1):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "at least one"):
                make_document_windows([[1, 2, 3]], width)

    def test_matches_previous_sentence_local_window_order(self):
        documents = [[1, 2, 3, 4, 5, 6, 7], [9, 8, 7, 6, 5], [0, 1]]
        previous_rows = [
            row for ids in documents
            for row in zip(*(ids[offset:] for offset in range(5)))
        ]
        np.testing.assert_array_equal(make_document_windows(documents, 5), previous_rows)

    def test_each_objective_uses_the_same_raw_windows_differently(self):
        windows = torch.tensor(make_document_windows([[1, 2, 3, 4, 5, 6]], 5))
        cbow_inputs, cbow_answers = objective_pairs(windows, "CBOW", bos_id=9)
        self.assertEqual(cbow_inputs.tolist(), [[1, 2, 4, 5], [2, 3, 5, 6]])
        self.assertEqual(cbow_answers.tolist(), [3, 4])
        skip_inputs, skip_answers = objective_pairs(windows, "Skip-gram", bos_id=9)
        self.assertEqual(skip_inputs.tolist(), [3, 3, 3, 3, 4, 4, 4, 4])
        self.assertEqual(skip_answers.tolist(), [1, 2, 4, 5, 2, 3, 5, 6])
        sources, targets = objective_pairs(windows, "Encoder-decoder", bos_id=9)
        self.assertEqual(sources.tolist(), [[1, 2, 3], [2, 3, 4]])
        self.assertEqual(targets.tolist(), [[9, 4, 5], [9, 5, 6]])


if __name__ == "__main__":
    unittest.main()
