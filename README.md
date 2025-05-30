## Emergent Misalignment Refusal Repo

This short research sprint was completed as part of the inaugural edition of the Technical Alignment Research Accelerator (TARA) in May 2025.

For this project, I investigated refusals in emergent misalignment for the Qwen2.5-coder-32b-instruct model. In the original paper, they reported increased refusals in gpt-4o on benign questions for their insecure misaligned model. An open question remained whether this would also be seen in other models.

My three main goals with this project were:
1. To determine if increased refusals occurred with Qwen2.5-coder-32b-instruct, the large open source model they used in the paper.
2. To explore what behaviour occurred if the refusal direction was removed and then the model was run on evaluations.
3. To replicate some of the emergent misalignment paper results, finetune my own model, and extend their investigations by testing the model on the evaluation XsTest

## Main Results

The main results will soon be reported in a short blog post summarising my 

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

### References
Inspired by and using code from the paper and codebases for the following research:
   @misc{betley2025emergentmisalignmentnarrowfinetuning,
        title={Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs},
        author={Jan Betley and Daniel Tan and Niels Warncke and Anna Sztyber-Betley and Xuchan Bao and Martín Soto and Nathan
        Labenz and Owain Evans},
        year={2025},
        eprint={2502.17424},
        archivePrefix={arXiv},
        primaryClass={cs.CR},
        url={https://arxiv.org/abs/2502.17424},
    }
    https://github.com/emergent-misalignment/emergent-misalignment

    @article{arditi2024refusal,
        title={Refusal in Language Models Is Mediated by a Single Direction},
        author={Andy Arditi and Oscar Obeso and Aaquib Syed and Daniel Paleka and Nina Panickssery and Wes Gurnee and Neel Nanda},
        journal={arXiv preprint arXiv:2406.11717},
        year={2024}
    }
    https://github.com/andyrdt/refusal_direction/tree/main

