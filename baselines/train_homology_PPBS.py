from baselines import predict_homology
from utilities.paths import library_folder,homology_folder
from utilities import dataset_utils,io_utils
import os
import pandas as pd
from baselines.train_handcrafted_features_PPBS import make_PR_curves
import numpy as np

check = True
train = True
fresh = False

ncores = 20
if check:
    ntemplates_max = 100
    nexamples_max = 5
    nhits_cut = 100
else:
    nexamples_max = 100
    ntemplates_max = None
    nhits_cut = 1000

homology_model = predict_homology.HomologyModel(predict_homology.template_databases['interface'], ncores=ncores, name=predict_homology.names['interface'],ntemplates_max=ntemplates_max,biounit=predict_homology.biounits['interface'])

if train:
    homology_model.fit(predict_homology.train_databases['interface'], nexamples_max=nexamples_max, nhits_cut=1000)

test_sets = [library_folder + 'datasets/PPBS/%s'%x for x in ['labels_test_70.txt','labels_test_homology.txt','labels_test_none.txt','labels_test_topology.txt'] ]
result_file = homology_folder + 'predictions_homology_PPBS.data'

if (not fresh) & os.path.exists(result_file):
    env = io_utils.load_pickle(result_file)
    completed_origins = env['completed_origins']
    completed_sequences = env['completed_sequences']
    completed_resids = env['completed_resids']
    completed_labels = env['completed_labels']
    completed_predictions = env['completed_predictions']
    completed_set = env['completed_set']
else:
    env = {}
    completed_origins = []
    completed_sequences = []
    completed_resids = []
    completed_labels = []
    completed_predictions = []
    completed_set = []


count = 0

for test_set in test_sets:
    subset = test_set.split('_')[-1][:-4]
    test_origins, test_sequences, test_resids, test_labels = dataset_utils.read_labels(test_set)
    if check:
        test_origins = test_origins[:10]
        test_sequences = test_sequences[:10]
        test_resids = test_resids[:10]
        test_labels = test_labels[:10]

    for test_origin,test_sequence,test_resid,test_label in zip(test_origins, test_sequences, test_resids, test_labels):
        already_done = test_origin in completed_origins
        if not already_done:
            prediction, sequence, resid = homology_model.predict(test_origin,biounit=True)
            completed_origins.append(test_origin)
            completed_sequences.append(sequence)
            completed_resids.append(resid)
            completed_labels.append(test_labels)
            completed_predictions.append(prediction)
            completed_set.append(subset)
            count += 1
        if count % 50 == 1:
            env = {
                'completed_origins': completed_origins,
                'completed_sequences': completed_sequences,
                'completed_resids': completed_resids,
                'completed_labels': completed_labels,
                'completed_predictions': completed_predictions,
                'completed_set': completed_set,
            }
            io_utils.save_pickle(env,result_file)


dataset_table = pd.read_csv(library_folder+'datasets/PPBS/table.csv')
completed_weights = []
for origin in completed_origins:
    completed_weights.append(  dataset_table['Sample weight'][ dataset_table['origin']==origin][0] )

env['completed_weights'] = completed_weights
for key,item in env.items():
    env[key] = np.array(item)
io_utils.save_pickle(env,result_file)


env = io_utils.load_pickle(result_file)
completed_origins = env['completed_origins']
completed_sequences = env['completed_sequences']
completed_resids = env['completed_resids']
completed_labels = env['completed_labels']
completed_predictions = env['completed_predictions']
completed_set = env['completed_set']

subset_masks = [
    completed_set == '70',
    completed_set == 'homology',
    completed_set == 'topology',
    completed_set == 'none',

]

fig = make_PR_curves(
    [completed_labels[mask] for mask in subset_masks],
    [completed_predictions[mask] for mask in subset_masks],
    [completed_weights[mask] for mask in subset_masks],
    ['Test (70%)','Test (Homology)','Test (Topology)','Test (None)'],
    title = 'Protein-protein binding site prediction: homology',
    figsize=(10, 10),
    margin=0.05,grid=0.1
    ,fs=25)

fig.savefig(library_folder + 'plots/PR_curve_PPBS_%s.png' % ('homology_check' if check else 'homology'), dpi=300)