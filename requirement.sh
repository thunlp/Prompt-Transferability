pip3 install -r requirement.txt

#Download GLUE
git clone git@github.com:nyu-mll/GLUE-baselines.git
python3 download_glue_data.py --data_dir glue_data --tasks all
mv GLUE-baselines data

#Need

#BertForMaskedLM
#RobertaForMaskedLM
#RobertaLargeForMaskedLM
#task_prompt_emb
#data
