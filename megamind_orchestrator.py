import json
import yaml
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from agent import create_agent


class MegamindOrchestrator:
    
    STAGES = {
        'question_generation': 'Generating specialized questions',
        'parallel_research': 'Running parallel research agents',
        'first_synthesis': 'Synthesizing first draft',
        'validation': 'Running validation agents',
        'final_synthesis': 'Creating final answer'
    }
    
    def __init__(self, config_path="config.yaml", silent=False):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.task_timeout = self.config['orchestrator']['task_timeout']
        self.silent = silent
        
        self.stage_progress = {}
        self.agent_progress = {}
        self.progress_lock = threading.Lock()
        
        self._init_megamind_config()
    
    def _init_megamind_config(self):
        megamind = self.config.get('megamind', {})
        
        self.question_prompt = megamind.get('question_generation_prompt', '''
You are a strategic question generator. Given a user query, generate exactly 4 specialized questions that will help gather comprehensive information.

Original query: {user_input}

Generate 4 questions, each from a different angle:
1. Research-focused: Gather factual information and data
2. Analysis-focused: Examine implications, patterns, and deeper meaning
3. Alternatives-focused: Explore different approaches, options, or perspectives
4. Verification-focused: Cross-check facts, identify potential issues or contradictions

Return ONLY a JSON array of 4 strings:
["research question", "analysis question", "alternatives question", "verification question"]
''')
        
        self.synthesis_prompt = megamind.get('synthesis_prompt', '''
You have received responses from 4 specialized research agents analyzing a query from different angles.

AGENT RESPONSES:
{agent_responses}

Synthesize these into a comprehensive first draft that combines insights from all perspectives.
Focus on creating a cohesive, well-organized response.
Do not mention that this is a synthesis or draft.
''')
        
        self.validation_prompt_1 = megamind.get('validation_prompt_1', '''
You are Validation Agent 1: Accuracy Checker.

ORIGINAL USER QUERY:
{original_query}

DRAFT ANSWER:
{draft_answer}

Your task:
1. Verify the draft actually answers the original question
2. Check for factual accuracy and consistency
3. Identify any gaps or missing information
4. Suggest specific improvements

Provide a structured critique with:
- Accuracy score (1-10)
- List of issues found
- Suggested improvements
''')
        
        self.validation_prompt_2 = megamind.get('validation_prompt_2', '''
You are Validation Agent 2: Quality Reviewer.

ORIGINAL USER QUERY:
{original_query}

DRAFT ANSWER:
{draft_answer}

Your task:
1. Evaluate clarity and organization
2. Check for completeness and depth
3. Assess if the response is actionable and useful
4. Suggest improvements for readability

Provide a structured critique with:
- Quality score (1-10)
- Strengths of the response
- Areas for improvement
''')
        
        self.final_synthesis_prompt = megamind.get('final_synthesis_prompt', '''
You are the Final Synthesis Agent. Your job is to create the definitive answer.

ORIGINAL USER QUERY:
{original_query}

FIRST DRAFT:
{draft_answer}

VALIDATION AGENT 1 FEEDBACK:
{validation_1}

VALIDATION AGENT 2 FEEDBACK:
{validation_2}

Create the final, polished answer.

STRUCTURE YOUR RESPONSE EXACTLY AS FOLLOWS:

## Executive Summary
[Provide a concise 2-3 sentence summary of the answer]

## Final Answer
[Provide the complete but focused answer. Integrate the validation feedback to improve accuracy and clarity. Avoid unnecessary verbosity or showing the step-by-step working unless asked.]

IMPORTANT:
- Do NOT output any JSON at the end.
- Do NOT mention "Validation Agent" or "Draft" in the final text.
- Start directly with the header "## Executive Summary".
''')
    
    def update_stage(self, stage: str, status: str):
        with self.progress_lock:
            self.stage_progress[stage] = status
    
    def update_agent_progress(self, agent_id: str, status: str, result: str = None):
        with self.progress_lock:
            self.agent_progress[agent_id] = {'status': status, 'result': result}
    
    def get_progress_status(self) -> Dict[str, Any]:
        with self.progress_lock:
            return {
                'stages': self.stage_progress.copy(),
                'agents': self.agent_progress.copy()
            }
    
    def _create_agent(self):
        return create_agent(silent=True)
    
    def _run_agent(self, agent_id: str, prompt: str, remove_tools: bool = True) -> Dict[str, Any]:
        try:
            self.update_agent_progress(agent_id, "PROCESSING")
            
            agent = self._create_agent()
            if remove_tools:
                for tool_name in list(agent.tool_mapping.keys()):
                    if tool_name != 'search' and tool_name != 'calculator':
                        agent.remove_tool(tool_name)
            
            start_time = time.time()
            response = agent.run(prompt)
            execution_time = time.time() - start_time
            
            self.update_agent_progress(agent_id, "COMPLETED", response)
            
            return {
                "agent_id": agent_id,
                "status": "success",
                "response": response,
                "execution_time": execution_time
            }
        except Exception as e:
            self.update_agent_progress(agent_id, "FAILED")
            return {
                "agent_id": agent_id,
                "status": "error",
                "response": f"Error: {str(e)}",
                "execution_time": 0
            }
    
    def generate_questions(self, user_input: str) -> List[str]:
        self.update_stage('question_generation', 'IN_PROGRESS')
        self.update_agent_progress('question_gen', 'PROCESSING')
        
        agent = self._create_agent()
        agent.remove_tool('mark_task_complete')
        
        prompt = self.question_prompt.format(user_input=user_input)
        
        try:
            response = agent.run(prompt)
            questions = json.loads(response.strip())
            
            if len(questions) != 4:
                raise ValueError(f"Expected 4 questions, got {len(questions)}")
            
            self.update_agent_progress('question_gen', 'COMPLETED', response)
            self.update_stage('question_generation', 'COMPLETED')
            return questions
            
        except (json.JSONDecodeError, ValueError):
            fallback = [
                f"Research comprehensive information about: {user_input}",
                f"Analyze and provide insights about: {user_input}",
                f"Find alternative perspectives on: {user_input}",
                f"Verify and cross-check facts about: {user_input}"
            ]
            self.update_agent_progress('question_gen', 'COMPLETED', str(fallback))
            self.update_stage('question_generation', 'COMPLETED')
            return fallback
    
    def run_parallel_agents(self, questions: List[str], images: Optional[List[dict]] = None) -> List[Dict[str, Any]]:
        self.update_stage('parallel_research', 'IN_PROGRESS')
        
        agent_names = ['research', 'analysis', 'alternatives', 'verification']
        
        for name in agent_names:
            self.update_agent_progress(name, 'QUEUED')
        
        results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for i, (name, question) in enumerate(zip(agent_names, questions)):
                future = executor.submit(self._run_agent, name, question, False)
                futures[future] = name
            
            for future in as_completed(futures, timeout=self.task_timeout):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    name = futures[future]
                    results.append({
                        "agent_id": name,
                        "status": "error",
                        "response": f"Agent failed: {str(e)}",
                        "execution_time": 0
                    })
        
        self.update_stage('parallel_research', 'COMPLETED')
        return results
    
    def first_synthesis(self, agent_results: List[Dict[str, Any]]) -> str:
        self.update_stage('first_synthesis', 'IN_PROGRESS')
        self.update_agent_progress('synthesis', 'PROCESSING')
        
        agent_responses_text = ""
        agent_names = ['Research', 'Analysis', 'Alternatives', 'Verification']
        
        for result in agent_results:
            agent_id = result.get('agent_id', '')
            idx = ['research', 'analysis', 'alternatives', 'verification'].index(agent_id) if agent_id in ['research', 'analysis', 'alternatives', 'verification'] else 0
            name = agent_names[idx] if idx < len(agent_names) else agent_id
            agent_responses_text += f"=== {name.upper()} AGENT ===\n{result.get('response', '')}\n\n"
        
        prompt = self.synthesis_prompt.format(agent_responses=agent_responses_text)
        
        agent = self._create_agent()
        for tool_name in list(agent.tool_mapping.keys()):
            agent.remove_tool(tool_name)
        
        try:
            response = agent.run(prompt)
            self.update_agent_progress('synthesis', 'COMPLETED', response)
            self.update_stage('first_synthesis', 'COMPLETED')
            return response
        except Exception as e:
            self.update_agent_progress('synthesis', 'FAILED')
            self.update_stage('first_synthesis', 'FAILED')
            raise e
    
    def run_validation(self, original_query: str, draft_answer: str) -> List[Dict[str, Any]]:
        self.update_stage('validation', 'IN_PROGRESS')
        
        self.update_agent_progress('validator_1', 'QUEUED')
        self.update_agent_progress('validator_2', 'QUEUED')
        
        prompt_1 = self.validation_prompt_1.format(
            original_query=original_query,
            draft_answer=draft_answer
        )
        prompt_2 = self.validation_prompt_2.format(
            original_query=original_query,
            draft_answer=draft_answer
        )
        
        results = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_1 = executor.submit(self._run_agent, 'validator_1', prompt_1, True)
            future_2 = executor.submit(self._run_agent, 'validator_2', prompt_2, True)
            
            for future in as_completed([future_1, future_2], timeout=self.task_timeout):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "agent_id": "validator",
                        "status": "error",
                        "response": f"Validation failed: {str(e)}",
                        "execution_time": 0
                    })
        
        self.update_stage('validation', 'COMPLETED')
        return results
    
    def final_synthesis(self, original_query: str, draft_answer: str, validation_results: List[Dict[str, Any]]) -> str:
        self.update_stage('final_synthesis', 'IN_PROGRESS')
        self.update_agent_progress('final', 'PROCESSING')
        
        validation_1 = next((r['response'] for r in validation_results if r.get('agent_id') == 'validator_1'), '')
        validation_2 = next((r['response'] for r in validation_results if r.get('agent_id') == 'validator_2'), '')
        
        prompt = self.final_synthesis_prompt.format(
            original_query=original_query,
            draft_answer=draft_answer,
            validation_1=validation_1,
            validation_2=validation_2
        )
        
        agent = self._create_agent()
        for tool_name in list(agent.tool_mapping.keys()):
            agent.remove_tool(tool_name)
        
        try:
            response = agent.run(prompt)
            self.update_agent_progress('final', 'COMPLETED', response)
            self.update_stage('final_synthesis', 'COMPLETED')
            return response
        except Exception as e:
            self.update_agent_progress('final', 'FAILED')
            self.update_stage('final_synthesis', 'FAILED')
            raise e
    
    def orchestrate(self, user_input: str, images: Optional[List[dict]] = None, status_callback=None) -> Dict[str, Any]:
        self.stage_progress = {}
        self.agent_progress = {}
        
        all_results = {
            'questions': [],
            'research_results': [],
            'first_draft': '',
            'validation_results': [],
            'final_result': ''
        }
        
        if status_callback:
            status_callback(self.STAGES['question_generation'])
        questions = self.generate_questions(user_input)
        all_results['questions'] = questions
        
        if status_callback:
            status_callback(self.STAGES['parallel_research'])
        research_results = self.run_parallel_agents(questions, images)
        all_results['research_results'] = research_results
        
        if status_callback:
            status_callback(self.STAGES['first_synthesis'])
        first_draft = self.first_synthesis(research_results)
        all_results['first_draft'] = first_draft
        
        if status_callback:
            status_callback(self.STAGES['validation'])
        validation_results = self.run_validation(user_input, first_draft)
        all_results['validation_results'] = validation_results
        
        if status_callback:
            status_callback(self.STAGES['final_synthesis'])
        final_result = self.final_synthesis(user_input, first_draft, validation_results)
        all_results['final_result'] = final_result
        
        return all_results
