#!/bin/bash

#logfile="output-2025-05-29.log"

#ollama start --model llama3.1 --port 11434 --gpu 0 --memory 16G --threads 8

ollama start > /dev/null 2>&1 &

#echo "Logging to $logfile"
echo "Execution started at $(date)" > output-29may25.log #"output-2025-05-29.log"

# Run script manually and redirect output

# echo "Running llm_blender_eval.py..." | tee -a "output-2025-05-29.log"
# python llm_blender_eval.py >> "$logfile" 2>&1
# echo "Finished llm_blender_eval.py." | tee -a "output-2025-05-29.log"
# echo "--------------------------------------" >> "output-2025-04-12.log"

# echo "Running llm_blender_eval_1.py..." | tee -a "output-2025-04-11.log
# python llm_blender_eval_1.py >> "output-2025-05-29.log" 2>&1
# echo "Finished llm_blender_eval_1.py." | tee -a "output-2025-05-29.log"
# echo "--------------------------------------" >> "output-2025-05-29.log"


echo "Running llm_blender_eval_2.py..." | tee -a "output-2025-06-1.log"
python llm_blender_eval_2.py >> "$logfile" 2>&1
echo "Finished llm_blender_eval_2.py." | tee -a "output-2025-06-1-2.log"
echo "--------------------------------------" >> "output-2025-06-1-3.log"

echo "Running llm_router_kEqual1_eval.py..." | tee -a output-2025-06-1-router.log"
python llm_router_kEqual1_eval.py >> "$logfile" 2>&1
echo "Finished llm_router_kEqual1_eval.py." | tee -a "output-2025-06-1-router1.log"
echo "--------------------------------------" >> output-2025-06-1.log

echo "Running plot.py..." | tee -a output-2025-06-1-plot.log
python plot.py >> output-2025-06-1-plot.log 2>&1
echo "Finished plot.py." | tee -a output-2025-06-1-plot.log
echo "--------------------------------------" >> output-2025-06-1-plot.log

echo "Running scoring.py..." | tee -a eval.log
python scoring.py >> eval.log 2>&1
echo "Finished scoring.py." | tee -a eval.log
echo "--------------------------------------" >> eval.log

echo "Execution finished at $(date)" >> eval.log
