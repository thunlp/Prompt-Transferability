#rm -rf task_prompt_emb/*

'''
echo Do you wanna rewrite task emb in the task_prompt_emb y/n ?
read ans

if [ $ans = "y" ]
then
    rm -rf task_prompt_emb/*
else
    echo "Do not rewrite"
    #exit
fi
'''


gpus=2
#CUDA_VISIBLE_DEVICES=$gpus

for i in {1..15}
do
    #echo $i
    #SST
    CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/SST2PromptRoberta.config \
        --gpu $gpus \
        --checkpoint /data3/private/suyusheng/prompt/prompt/model/SST2PromptRoberta/$i.pkl \

    cd task_prompt_emb
    mv SST2PromptRoberta SST2PromptRoberta_$i
    cd ..
done


cd task_prompt_emb
mv SST2PromptRoberta_15 SST2PromptRoberta




