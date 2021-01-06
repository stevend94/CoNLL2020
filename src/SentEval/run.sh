# Run bash file to get SentEval results from experiments
# Move this file and eval.py to the SentEval folder

# Change to the folder location of the embeddings
PATH_TO_EMBEDDINGS=''

FILE=Results
if [ -f "$FILE" ]; then
    echo "$FILE found!"
else 
    echo "$FILE does not exist. Creating results folder"
    mkdir Results
fi

MODELS=('word2vec.npy'
        'fasttext.npy'
        'glove.npy'
        'input_embeddings.npy'
        'output_embeddings.npy'
        'activation_embeddings.npy')

# Run SentEval experiments on all folders
echo "RUN SentEval experiments (Requires Pytorch)"
for model in ${MODELS[@]}; do
    python eval.py --path $PATH_TO_EMBEDDINGS$model
done