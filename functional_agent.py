# functional_agent.py

import asyncio
import logging

logger = logging.getLogger("autonomous_system.functional_agent")


class FunctionalAgent:
    """
    Orchestrates multi-phase processing using the Llama model manager,
    handling planning, execution, digesting, validating, and responding.
    """

    def __init__(self, llama_manager, state=None):
        self.llama_manager = llama_manager
        self.state = state or {}
        self.logger = logging.getLogger("autonomous_system.functional_agent")

    async def handle_request(self, prompt):
        """Handles a request through multiple phases."""
        self.logger.debug(f"Handling request: {prompt}")

        # Generate private notes
        self.logger.info("Generating private notes...")
        self.state["processing_phase"] = "Notes"
        private_notes = await self.llama_manager.generate_private_notes(prompt)

        # Phase 1: Planning
        self.logger.info("Phase 1: Planning...")
        self.state["processing_phase"] = "Planning"
        plan_result = await self.llama_manager.run_phase(
            "Planning", prompt, private_notes
        )

        # Phase 2: Execution
        self.logger.info("Phase 2: Execution...")
        self.state["processing_phase"] = "Execution"
        execution_result = await self.llama_manager.run_phase(
            "Execution", prompt, plan_result
        )

        # Phase 3: Digesting
        self.logger.info("Phase 3: Digesting...")
        self.state["processing_phase"] = "Digesting"
        digest_result = await self.llama_manager.run_phase(
            "Digesting", prompt, execution_result
        )

        # Phase 4: Validating
        self.logger.info("Phase 4: Validating...")
        self.state["processing_phase"] = "Validating"
        validate_result = await self.llama_manager.run_phase(
            "Validating", prompt, digest_result
        )

        # Phase 5: Responding
        self.logger.info("Phase 5: Responding...")
        self.state["processing_phase"] = "Responding"
        final_response = await self.llama_manager.run_phase(
            "Responding", prompt, validate_result
        )

        self.logger.debug(f"Final response: {final_response}")
        self.logger.info("Request processing complete.")
        self.state["processing_phase"] = "Done"
        return final_response
