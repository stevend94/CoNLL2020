model_paths=('../DSMs/word2vec.npy'
             '../DSMs/fasttext.npy'
             '../DSMs/glove.npy'
             '../DSMs/input_embeddings.npy'
             '../DSMs/output_embeddings.npy'
             '../DSMs/activation_embeddings.npy')
             
nnlm_modes=('input'
            'output'
            'tied')

FILE=data/PennTreeBank
if [ -f "$FILE" ]; then
    echo "$FILE found!"
else 
    echo "$FILE does not exist. Building processed data"
    mkdir data
    mkdir History
    
    cd data
    mkdir PennTreeBank
    cd ..
    
    python LanguageModel.py --build True --check_gpu True --dsm_path '../DSMs/input_embeddings.npy' --data_path 'data/PennTreeBank/processed'
fi

for model_path in ${model_paths[@]}; do
    for mode_type in ${nnlm_modes[@]}; do    
        python LanguageModel.py --dsm_path $model_path --model $mode_type 
    done 
done    
        
    