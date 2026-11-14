"""The authorized78GiB cap must not retain the older90-percent restriction."""
import copy,json,tempfile,unittest
from pathlib import Path
from runner import ROOT
from runner.worker import gpu_allocation_fraction
from runner.supervisor import settings


class GPUMemoryPolicy(unittest.TestCase):
    def test_effective_cap_is78gib_on80gib_card(self):
        fraction=gpu_allocation_fraction(80*(1<<30),78)
        self.assertEqual(fraction,.975)
        self.assertEqual(fraction*80,78)
        self.assertGreater(fraction,.90)
    def test_settings_accept78_and_reject_above_authorized_cap(self):
        value=json.loads((ROOT/'runtime/settings.json').read_text());self.assertEqual(value['maximum_gpu_allocation_gib'],78)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'settings.json';p.write_text(json.dumps(value));self.assertEqual(settings(p)['maximum_gpu_allocation_gib'],78)
            value['maximum_gpu_allocation_gib']=79;p.write_text(json.dumps(value))
            with self.assertRaises(AssertionError):settings(p)
        with self.assertRaises(ValueError):gpu_allocation_fraction(80*(1<<30),79)


if __name__=='__main__':unittest.main()
