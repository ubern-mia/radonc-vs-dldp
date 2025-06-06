import unittest
import subprocess
import os
import glob


class TestGenerateFigures(unittest.TestCase):
    def test_generate_figure_scripts(self):
        # Adjust the pattern to match your script naming convention and directory
        script_pattern = os.path.join("generate_figure_*.py")
        scripts = glob.glob(script_pattern)
        self.assertTrue(scripts, f"No scripts found matching pattern: {script_pattern}")
        for script in scripts:
            with self.subTest(script=script):
                result = subprocess.run(
                    ["python", script], capture_output=True, text=True
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    f"{script} failed with error:\n{result.stderr}",
                )


if __name__ == "__main__":
    unittest.main()
