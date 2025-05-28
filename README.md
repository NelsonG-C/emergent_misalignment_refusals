## Finetuning steps

Run these commands:
'''bash
wandb login
'''

Set up your .env file in the VastAI instance. You'll need an OPENAI_API_KEY and a Huggingface API key

## Evaluation Steps

### Part One: Replication of results and investigating Refusals for emergent misalignment

Each of the models (control, em-insecure, and finetuned-custom-insecure) are run on three evaluations:
- em_refusals_main_questions: The main misaligned questions from the paper, used for the replication comparison
- deception: Also from the paper, used to measure some refusals and for replication
- xstest: An evaluation to test a models ability to refuse on safe and unsafe prompts

To produce the evaluations, do the following for each of the models:
- Install the packages with ```pip install -r requirements.txt```
- Update the run_evaluation.py file with arguments for the eval
    - Main args are model name, batch size, subset size, and setting which evals to run
- Run ```python evals/run_evaluation```

### Part Two: Removing refusals and evaluating changes

The refusal directions paper has a repository which we are using to apply the refusal direction ablation to all of our models.
This will allow us to see what the affect of the increased refusals may be doing when we remove them.

