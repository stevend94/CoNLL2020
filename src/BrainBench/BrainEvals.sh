#!/usr/bin/env python

if [ $# -eq 0 ]; then
  # used if no arguments given
  echo "No arguments provided so using default values"
  MODEL_PATHS=('../DSMs/word2vec.npy'
#              '../DSMs/fasttext.npy'
#              '../DSMs/glove.npy'
             '../DSMs/output_embeddings.npy'
             '../DSMs/input_embeddings.npy'
             '../DSMs/activation_embeddings.npy')
    
else
  # else use arguments
  MODEL_PATHS=( "$@" )
fi

for model_path in ${MODEL_PATHS[@]}; do
  echo $model_path
  start=`date +%s`

  filtered_path=$(python FilterModel.py --model $model_path)
  chmod +x $model_path

  new_filt_path=$(python getFiltPath.py $filtered_path)
  python filterVectors.py data/Brainbench_words.txt < $filtered_path > $new_filt_path
  python BrainEval.py $new_filt_path

  end=`date +%s`
  echo $((end-start))
done
