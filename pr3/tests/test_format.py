import unittest
import aircraft_eval as ae

class TestFormat(unittest.TestCase):
    def test_recovery(self):
        raw='{"label":"CIVILIAN","reason":"The aircraft is painted with the "\n    '
        self.assertEqual(ae.parse_label(raw),'civilian')
        self.assertFalse(ae.is_valid_json_response(raw))
        self.assertIsNone(ae.parse_label('military or civilian'))
    def test_valid_json(self):
        raw='{"label":"MILITARY","reason":"insignia"}'
        self.assertEqual(ae.parse_label(raw),'military')
        self.assertTrue(ae.is_valid_json_response(raw))
if __name__=='__main__':unittest.main()
