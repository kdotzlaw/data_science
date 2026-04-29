import shared_utils as su
import logging, os
from src import eda
from src import diffex

# test shared_utils
exp, sample, ann = su.load_geo_dataset('GSE19804')
print(exp.shape, sample.shape,ann.shape)
print(su.detect_log_scale(exp))
samp = su.assign_groups(sample,source_col='source_name_ch1',
                        substrings={'tumor':'tumor','normal':'normal'})
print(samp['group'].value_counts())
de = su.differential_expression(su.normalize(exp,'log2'),samp, 'tumor','normal',ann)
print(de.head())

# test eda
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
out = os.path.join('result', 'GSE19804', 'eda')
eda.run_eda(exp, samp, out)
print('outputs:', sorted(os.listdir(out)))

# test diffex
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
out = os.path.join('result', 'GSE19804')
de = diffex.run_diffex(exp, samp, ann, 'tumor', 'normal', output_dir=out)
print('shape:', de.shape, 'cols:', list(de.columns))
print('csv exists:', os.path.exists(os.path.join(out, 'de.csv')))
