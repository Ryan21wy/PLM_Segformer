import requests
import zipfile
import unittest
from inference.model_prediction import *


def download(url, fname):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(desc=fname, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


class test(unittest.TestCase):
    def setUp(self):
        dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(dirname, "data\imgs")
        self.label_path = os.path.join(dirname, "data\labels")
        self.model_path = os.path.join(dirname, "PLM_Segformer_models.zip")
        self.model_dir = os.path.join(dirname, "PLM_Segformer_models")
        if not(os.path.isfile(self.model_path)):
            url = 'https://github.com/Ryan21wy/PLM_Segformer/releases/download/v1.0/PLM_Segformer_models.zip'
            download(url, self.model_path)
            zip_file = zipfile.ZipFile(self.model_path)
            zip_file.extractall('PLM_Segformer_models')
            zip_file.close()

    def test_hpic(self):
        pred, Hits, IoUs = prediction(self.data_path, self.model_dir, label_path=self.label_path,
                                      n_class=2, crop_size=(512, 768), TTA=False, TLC=False, post=False)
        self.assertListEqual(IoUs, [0.859, 0.673, 0.468, 0.805, 0.967])


if __name__ == '__main__':
    unittest.main()
