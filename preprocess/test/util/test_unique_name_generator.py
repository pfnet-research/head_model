import unittest

from lib.util import unique_name_generator


class TestUniqueNameGenerator(unittest.TestCase):

    def setUp(self):
        self.gen = unique_name_generator.UniqueNameGenerator()

    def test_duplicate(self):
        self.assertEqual(self.gen.make_unique('foo'),
                         'foo')
        self.assertEqual(self.gen.make_unique('foo'),
                         'foo_1')
        self.assertEqual(self.gen.make_unique('foo'),
                         'foo_2')

    def test_different(self):
        self.gen.make_unique('foo')
        self.gen.make_unique('foo')
        self.assertEqual(self.gen.make_unique('bar'),
                         'bar')
        self.assertEqual(self.gen.make_unique('bar'),
                         'bar_1')

    def test_space(self):
        self.assertEqual(self.gen.make_unique('f o o'),
                         'f o o')
        self.assertEqual(self.gen.make_unique('f o o'),
                         'f o o_1')
