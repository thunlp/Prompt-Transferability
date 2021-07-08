gpus=7

python3 test.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --checkpoint  model/SST2PromptRoberta/15.pkl \
    --result model/SST2PromptRoberta_result \
    #--data_type eval \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed
