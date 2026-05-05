import shared_utils as su
import logging, os
import pandas as pd
from shared_utils import load_geo_dataset, assign_groups
from src import eda, diffex, volcano, heatmap,compare

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

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
'''logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
out = os.path.join('result', 'GSE19804', 'eda')
eda.run_eda(exp, samp, out)
print('outputs:', sorted(os.listdir(out)))
'''

# test diffex
'''logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
out = os.path.join('result', 'GSE19804')
de = diffex.run_diffex(exp, samp, ann, 'tumor', 'normal', output_dir=out)
print('shape:', de.shape, 'cols:', list(de.columns))
print('csv exists:', os.path.exists(os.path.join(out, 'de.csv')))
'''

# test volcano
'''logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
de = pd.read_csv(os.path.join('result', 'GSE19804', 'de.csv'))
out_png = os.path.join('result', 'GSE19804', 'volcano.png')
volcano.plot_volcano(de, out_png)
print('volcano exists:', os.path.exists(out_png))
'''

# test heatmap
'''logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
expression, samples, annotation = load_geo_dataset('GSE19804', cache_dir='data')
samples = assign_groups(
    samples,
    source_col='characteristics_ch1',
    substrings={'tumor': 'tumor', 'normal': 'normal'},
)
de = pd.read_csv(os.path.join('result', 'GSE19804', 'de.csv'))
out_png = os.path.join('result', 'GSE19804', 'heatmap.png')
heatmap.plot_heatmap(expression, samples, de, out_png, top=50)
print('heatmap exists:', os.path.exists(out_png))
'''

# test compare
def collapsed_de(accession, tumor_substring='tumor'):
    expression, samples, annotation = load_geo_dataset(accession, cache_dir='data')
    samples = assign_groups(
        samples,
        source_col='source_name_ch1',
        substrings={'tumor': tumor_substring, 'normal': 'normal'},
    )
    return diffex.run_diffex(
        expression, samples, annotation, 'tumor', 'normal',
        output_dir=None, collapse_to_gene=True, print_head=False,
    )

de_a = collapsed_de('GSE19804')
de_b = collapsed_de('GSE10072', tumor_substring='adenocarcinoma')
out = os.path.join('result', 'compare_GSE19804_vs_GSE10072')
joined = compare.run_compare(de_a, de_b, out,
                             label_a='GSE19804', label_b='GSE10072')
print('joined shape:', joined.shape)
for fname in ('shared_de_genes.csv', 'log2fc_scatter.png', 'overlap_summary.txt'):
    print(fname, 'exists:', os.path.exists(os.path.join(out, fname)))