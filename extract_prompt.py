import os
import subprocess
import sys
from unittest.mock import patch

gpu=2
except_token = None

python_file = "create_prompt.py"
source_code = open(python_file).read()
compiled = compile(source_code, filename=python_file, mode="exec")


prompt_model = os.listdir("config")


prompt_model = os.listdir("model")
prompt_model = [name for name in prompt_model if ".py" not in name and '__pycache__' not in name]
print(prompt_model)


for file in prompt_model:
    all_pkl= os.listdir("model/"+file)

exit()

with patch('sys.argv', [python_file, "2"]):
    exec(compiled)

#exec(open("del.py").read(), {"gpu":2})
#exec(open("del.py").read(),"gpu":2)

#subprocess.call([sys.executable,'extract_prompt.py', '2'])
#execfile("del.py")


