echo "RETRIEVING ONE BILLION WORD LANGUAGE MODEL"
python lm_1b_eval.py --mode get_data

echo "RETRIEVING LSTM MODEL WEIGHTS"
python lm_1b_eval.py --mode get_model

model_paths=('DSMs/word2vec.npy'
             'DSMs/fasttext.npy'
             'DSMs/glove.npy'
             'DSMs/input_embeddings.npy'
             'DSMs/max_dsm.npy'
             'DSMs/weight_dsm.npy')
             
             
# Retrieve input and output embeddings, and build DSMs
echo "EXTRACTING INPUT AND OUTPUT EMBEDDINGS"
python lm_1b_eval.py --mode dump_emb --save_dir "outputs" --vocab_file "data/vocab-2016-09-10.txt" --pbtxt "data/graph-2016-09-10.pbtxt" --ckpt "data/ckpt-*"
python utils.py --build True

# Train activation maximization embeddings
echo "TRAINING ACTIVATION MAXIMIZATION EMBEDDINGS"
python activation.py --weight_path "DSMs/output_embeddings.npy" -- bias_path "outputs/softmax_bias.npy" --save_path "DSMs/activation_embeddings.npy"

# Train other embedding models 
echo "TRAINING SOTA DISTRIBUTIONAL MODELS"
python embeddings.py --model "word2vec" --embedding_size 300 --context_window 5 --min_count 0 --workers -1
python embeddings.py --model "fasttext" --embedding_size 300 --context_window 5 --min_count 0 
python embeddings.py --model "glove" --embedding_size 300 --context_window 5 --learning_rate 0.05 --epochs 100 --workers -1 

# Run BrainBench
cd BrainBench 

FILE=corr_mats
if [ -f "$FILE" ]; then
    echo "$FILE found!"
else 
    echo "$FILE does not exist. Copying GitHub Repo"
    git clone https://github.com/haoyanxu/2v2_software.git
    
    cd 2v2_software
    mv corr_mats ..
    cd ..
fi

FILE=data
if [ -f "$FILE" ]; then
    echo "$FILE found!"
else 
    echo "$FILE does not exist. Creating data file"
    mkdir data
    cd data
    mkdir FilteredVectors
    cd ..
fi

bash BrainEvals.sh

cd ..


# write embeddings to experiment folder for intrinsic tasks
echo "RUN INTRINSIC EVALUATIONS (WORD SIMILARITY BENCHMARKS)"
for model_path in ${model_paths[@]}; do
    python utils.py --vecto_write True -p $model_path
done

python utils.py --vecto_bench True

# Neural Language Modelling
echo "PERFORMING NEURAL NETWORK LANGUAGE MODEL EXPERIMENTS"
cd NNLM
bash run_nnlm.sh
cd ..
