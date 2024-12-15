# functional_agent.py

import asyncio
import logging

logger = logging.getLogger("autonomous_system.functional_agent")


class FunctionalAgent:
    """
    Orchestrates multi-phase processing using the Llama model manager,
    handling planning, execution, digesting, validating, and responding.
    """

    def __init__(self, llama_manager):
        self.llama_manager = llama_manager
        self.logger = logging.getLogger("autonomous_system.functional_agent")

    async def handle_request(self, prompt):
        """Handles a request through multiple phases."""
        self.logger.debug(f"Handling request: {prompt}")

        # Generate private notes
        private_notes = await self.llama_manager.generate_private_notes(prompt)

        # Phase 1: Planning
        plan_result = await self.llama_manager.run_phase(
            "Planning", prompt, private_notes
        )

        # Phase 2: Execution
        execution_result = await self.llama_manager.run_phase(
            "Execution", prompt, plan_result
        )

        # Phase 3: Digesting
        digest_result = await self.llama_manager.run_phase(
            "Digesting", prompt, execution_result
        )

        # Phase 4: Validating
        validate_result = await self.llama_manager.run_phase(
            "Validating", prompt, digest_result
        )

        # Phase 5: Responding
        final_response = await self.llama_manager.run_phase(
            "Responding", prompt, validate_result
        )

        self.logger.debug(f"Final response: {final_response}")
        return final_response
