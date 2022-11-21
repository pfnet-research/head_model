import unittest

from lib.util import id_converter


class TestIDConverter(unittest.TestCase):

    def setUp(self):
        self.converter = id_converter.IDConverter()

    def test_empty(self):
        self.assertEqual(self.converter.id2name, [])
        self.assertDictEqual(self.converter.name2id, {})
        self.assertEqual(self.converter.unique_num, 0)

    def test_to_id(self):
        names = [10, 2, 10, 5]
        ids = [self.converter.to_id(name) for name in names]
        self.assertEqual(ids, [0, 1, 0, 2])
        self.assertEqual(self.converter.id2name, [10, 2, 5])
        self.assertDictEqual(self.converter.name2id, {10: 0, 2: 1, 5: 2})
        self.assertEqual(self.converter.unique_num, 3)

    def test_to_name(self):
        expected = [10, 2, 10, 5]
        ids = [self.converter.to_id(name) for name in expected]
        actual = [self.converter.to_name(id_) for id_ in ids]
        self.assertEqual(actual, expected)

    def test_non_existent(self):
        self.converter.to_id('foo')
        self.converter.to_id('bar')
        with self.assertRaises(ValueError):
            self.converter.to_name(3)
