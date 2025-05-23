import asyncio
from model_manager import get_llm_instance
from inspect_evaluations import run_inspect_evaluations_sync
from .evals import truthfulqa, xstest

async def main():
    # Load your model
    llm, _ = get_llm_instance()

    log = eval(truthfulqa(), model=llm, limit=10, log_dir=str("logs"))
    
    print(f"Evaluation complete!")

if __name__ == "__main__":
    asyncio.run(main())
