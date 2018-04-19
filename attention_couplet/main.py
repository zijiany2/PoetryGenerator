#! /usr/bin/env python
# -*- coding:utf-8 -*-

#from plan import Planner
from predict import Seq2SeqPredictor
from rhymes import get_rhyme, rhyme_boosting
import sys

import tensorflow as tf
tf.app.flags.DEFINE_boolean('cangtou', False, 'Generate Acrostic Poem')

reload(sys)
sys.setdefaultencoding('utf8')

def get_cangtou_keywords(input):
    assert(len(input) == 4)
    return [c for c in input]

def main(args, cangtou=False):

    #planner = Planner()
    with Seq2SeqPredictor() as predictor:
        # Run loop
        terminate = False
        while not terminate:
            try:
                input = raw_input('Input Text:\n').decode('utf-8').strip()

                if not input:
                    print 'Input cannot be empty!'
                elif input.lower() in ['quit', 'exit']:
                    terminate = True
                else:
                    lines = predictor.predict(input)
                    lines = lines.replace('UNK', '江')
                    lines = lines[:-1] + rhyme_boosting(lines[-1], get_rhyme(input[-1]))
                    print lines

            except EOFError:
                terminate = True
            except KeyboardInterrupt:
                terminate = True
    print '\nTerminated.'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nocouplet', help='No couplet', required=False)
    args = parser.parse_args()
    main(args, cangtou=tf.app.flags.FLAGS.cangtou)