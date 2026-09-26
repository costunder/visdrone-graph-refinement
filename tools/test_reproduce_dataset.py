import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reproduce_dataset as data


def sample(split, label='car', box=None, name='fixture.jpg'):
    return dict(filepath='data/' + name, tags=[split], metadata=dict(width=100, height=100),
                ground_truth=dict(detections=[dict(label=label, bounding_box=box or [.1, .2, .3, .4],
                                                  truncation=0, occlusion=1)]))


class DatasetTests(unittest.TestCase):
    def test_train_clipping_and_precision(self):
        _, labels, _ = data.convert([sample('train', box=[-.1, .2, .4, 1.])])
        self.assertEqual(labels['train/fixture.txt'], '3 0.15000000 0.60000000 0.30000000 0.80000000\n')

    def test_val_pixel_boxes_ids_and_precision(self):
        _, labels, gt = data.convert([sample('val', name='b.jpg'), sample('val', name='a.jpg')])
        self.assertEqual(labels['val/a.txt'], '3 0.250000 0.400000 0.300000 0.400000')
        self.assertEqual(gt['images'][0]['file_name'], 'a.jpg')
        self.assertEqual(gt['annotations'][0]['bbox'], [10., 20., 30., 40.])
        self.assertEqual(gt['annotations'][1]['image_id'], 2)
        self.assertEqual(gt['annotations'][0]['category_id'], 4)

    def test_ignored_and_unknown_classes(self):
        _, labels, gt = data.convert([sample('val', label='ignore_regions')])
        self.assertEqual(labels['val/fixture.txt'], '')
        self.assertEqual(gt['annotations'], [])
        with self.assertRaises(KeyError):
            data.convert([sample('train', label='unknown')])

    def test_nonfinite_fractional_val_and_overlap_rejected(self):
        for s in [sample('train', box=[float('nan'), 0, 1, 1]),
                  sample('val', box=[.123, 0, .1, .1])]:
            with self.assertRaises(ValueError):
                data.convert([s])
        with self.assertRaises(ValueError):
            data.convert([sample('train'), sample('val')])

    def test_unsafe_and_duplicate_paths(self):
        for name in ['../bad.jpg', '/data/bad.jpg', 'data/../bad.jpg', 'data/a\\b.jpg']:
            with self.assertRaises(ValueError):
                data.safe_name(name)
        with self.assertRaises(ValueError):
            data.convert([sample('train'), sample('train')])

    def test_verified_cache_never_requests_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'image.jpg'
            p.write_bytes(b'fixture')
            with patch.object(data.urllib.request, 'urlopen') as network:
                self.assertEqual(data.fetch('data/image.jpg', p, data.sha256(p)), p)
                with self.assertRaises(ValueError):
                    data.fetch('data/image.jpg', p, '0' * 64)
                network.assert_not_called()
            self.assertEqual(p.read_bytes(), b'fixture')

    def test_mock_download_checksum_and_cleanup(self):
        import io
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'image.jpg'
            digest = hashlib.sha256(b'image').hexdigest()
            with patch.object(data.urllib.request, 'urlopen', return_value=io.BytesIO(b'image')):
                data.fetch('data/image.jpg', p, digest)
            self.assertEqual(p.read_bytes(), b'image')
            self.assertEqual(list(Path(tmp).glob('*.part')), [])
            bad = Path(tmp) / 'bad.jpg'
            with patch.object(data.time, 'sleep'), patch.object(data.urllib.request, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(b'wrong')):
                with self.assertRaises(ValueError):
                    data.fetch('data/bad.jpg', bad, digest)
            self.assertFalse(bad.exists())
            self.assertEqual(list(Path(tmp).glob('*.part')), [])

    def test_existing_output_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(data.sys, 'argv', ['reproduce_dataset', '--download', '--output', tmp]), patch.object(data, 'fetch') as fetch:
                with self.assertRaises(FileExistsError):
                    data.main()
                fetch.assert_not_called()

    def test_semantic_json_hash_ignores_key_order(self):
        self.assertEqual(data.json_digest({'a': 1, 'b': 2}), data.json_digest({'b': 2, 'a': 1}))


if __name__ == '__main__':
    unittest.main()
