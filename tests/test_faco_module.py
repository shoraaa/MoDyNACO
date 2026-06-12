import unittest

import faco


class FacoModuleTests(unittest.TestCase):
    def test_thread_helpers_round_trip(self):
        original_threads = faco.get_faco_cpp_threads()
        target_threads = max(1, original_threads)

        try:
            faco.set_faco_cpp_threads(target_threads)
            self.assertEqual(faco.get_faco_cpp_threads(), target_threads)
        finally:
            faco.set_faco_cpp_threads(original_threads)

    def test_public_solver_symbols_are_importable(self):
        self.assertTrue(hasattr(faco, "MFACO_TSP"))
        self.assertTrue(hasattr(faco, "MFACO_CVRP"))
        self.assertTrue(hasattr(faco, "ACO_TSP"))
        self.assertTrue(hasattr(faco, "ACO_CVRP"))


if __name__ == "__main__":
    unittest.main()
