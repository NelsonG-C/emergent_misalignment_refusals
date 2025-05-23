import asyncio
from model_manager import get_llm_instance
from inspect_evaluations import run_inspect_evaluations_sync

async def main():
    # Load your model
    llm, _ = get_llm_instance()

    # This handles the event loop automatically
    filename, results = run_inspect_evaluations_sync(
        vllm_instance=llm,
        limit_per_eval=100
    )
    
    print(f"Evaluation complete! Results in {filename}")

if __name__ == "__main__":
    asyncio.run(main())
