import enum
import importlib.util
import sys
import types
import unittest


def load_time():
    serializable = types.ModuleType("composer.core.serializable")

    class Serializable:
        pass

    serializable.Serializable = Serializable
    utils = types.ModuleType("composer.utils")

    class StringEnum(str, enum.Enum):
        pass

    utils.StringEnum = StringEnum
    core = types.ModuleType("composer.core")
    core.serializable = serializable
    composer = types.ModuleType("composer")
    composer.core = core
    composer.utils = utils
    sys.modules["composer"] = composer
    sys.modules["composer.core"] = core
    sys.modules["composer.core.serializable"] = serializable
    sys.modules["composer.utils"] = utils
    spec = importlib.util.spec_from_file_location("composer.core.time", "composer/core/time.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["composer.core.time"] = module
    spec.loader.exec_module(module)
    return module.Time


class WctSecondsTest(unittest.TestCase):
    def test_a_bare_number_of_seconds_keeps_its_value(self):
        time = load_time()
        self.assertEqual(time.from_timedelta("5").value, 5)

    def test_minutes_and_seconds_keep_both(self):
        time = load_time()
        self.assertEqual(time.from_timedelta("1m10").value, 70)

    def test_a_marked_second_count_stays(self):
        time = load_time()
        self.assertEqual(time.from_timedelta("5s").value, 5)

    def test_one_minute_stays_sixty_seconds(self):
        time = load_time()
        self.assertEqual(time.from_timedelta("1m").value, 60)


if __name__ == "__main__":
    unittest.main()
