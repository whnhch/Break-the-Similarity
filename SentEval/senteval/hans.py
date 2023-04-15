# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
HANS- Entailment
'''
from __future__ import absolute_import, division, unicode_literals

import codecs
import os
import io
import copy
import logging
import numpy as np

from senteval.tools.validation import KFoldClassifier


class HANSEval(object):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : HANS Entailment*****\n\n')
        self.seed = seed
        train1 = self.loadFile(os.path.join(taskpath, 's1.train'))
        train2 = self.loadFile(os.path.join(taskpath, 's2.train'))

        trainlabels = io.open(os.path.join(taskpath, 'labels.train'),
                              encoding='utf-8').read().splitlines()

        test1 = self.loadFile(os.path.join(taskpath, 's1.test'))
        test2 = self.loadFile(os.path.join(taskpath, 's2.test'))
        testlabels = io.open(os.path.join(taskpath, 'labels.test'),
                             encoding='utf-8').read().splitlines()

        # sort data (by s2 first) to reduce padding
        sorted_train = sorted(zip(train2, train1, trainlabels),
                              key=lambda z: (len(z[0]), len(z[1]), z[2]))
        train2, train1, trainlabels = map(list, zip(*sorted_train))

        sorted_test = sorted(zip(test2, test1, testlabels),
                             key=lambda z: (len(z[0]), len(z[1]), z[2]))
        test2, test1, testlabels = map(list, zip(*sorted_test))

        self.samples = train1 + train2 + test1 + test2
        
        self.data = {'train': (train1, train2, trainlabels),
                     'test': (test1, test2, testlabels)
                     }

        
    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath):
        with open(fpath, 'r') as f:
            return [line.split() for line in f.read().splitlines()]
        
    def run(self, params, batcher):
        self.X, self.y = {}, {}
        self.X_t, self.y_t = {}, {}
        dico_label = {'entailment': 0,  'non-entailment': 1}
        for key in self.data:
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            input1, input2, mylabels = self.data[key]
            enc_input = []
            n_labels = len(mylabels)
            for ii in range(0, n_labels, params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)
                    enc_input.append(np.hstack((enc1, enc2, enc1 * enc2,
                                                np.abs(enc1 - enc2))))
                if (ii*params.batch_size) % (500*params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / n_labels))
            self.X[key] = np.vstack(enc_input)
            self.y[key] = [dico_label[y] for y in mylabels]

        config_classifier = {'nclasses': 2, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier,
                             'kfold': params.kfold}

        clf = KFoldClassifier({'X': self.X['train'],
                               'y': np.array(self.y['train'])},
                              {'X': self.X['test'],
                               'y': np.array(self.y['test'])},
                              config_classifier)
        devacc, testacc, _ = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1} for HANS\n'
                      .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ntest': len(self.data['test'][0])}