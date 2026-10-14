import unittest
from checkpoint_utils import validate_mapping, decode_character

class MappingTest(unittest.TestCase):
    def test_lexical_checkpoint_is_decoded_using_saved_order(self):
        mapping = {str(i): n for n, i in enumerate([1, 10, 2, 3, 4, 5, 6, 7, 8, 9])}
        inverse = validate_mapping(mapping, 10)
        self.assertEqual(decode_character(1, list("abcdefghij"), inverse), "j")
    def test_missing_duplicate_and_out_of_range_maps_are_rejected(self):
        for mapping in [None, {"1":0}, {"1":0,"2":0}, {"1":0,"3":1}]:
            with self.assertRaises(ValueError): validate_mapping(mapping, 2)
    def test_prediction_outside_map_is_rejected(self):
        with self.assertRaises(ValueError): decode_character(7, ["a"], {0:"1"})
