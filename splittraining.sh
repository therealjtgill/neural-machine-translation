#!/bin/bash

python3 nmttrainer.py -et ./tokenized_data_manythings_europarl_nolow/fr-en/tokenized.english -tt ./tokenized_data_manythings_europarl_nolow/fr-en/tokenized.french -ed ./tokenized_data_manythings_europarl_nolow/fr-en/dictionary.english -td ./tokenized_data_manythings_europarl_nolow/fr-en/dictionary.french -n 10 -m 30 -kp 1.0

latestfolder=$( ls ~/Documents/nmt_training_output/ -t | head -1 )
lastcheckpoint=$( ls ~/Documents/nmt_training_output/${latestfolder}/nmt_checkpoint* -d | head -1 )
checkpoint=${lastcheckpoint%.*}

python3 nmttrainer.py -et ./tokenized_data_manythings_europarl_nolow/fr-en/tokenized.english -tt ./tokenized_data_manythings_europarl_nolow/fr-en/tokenized.french -ed ./tokenized_data_manythings_europarl_nolow/fr-en/dictionary.english -td ./tokenized_data_manythings_europarl_nolow/fr-en/dictionary.french -n 10 -m 52 -c ${checkpoint} -kp 1.0