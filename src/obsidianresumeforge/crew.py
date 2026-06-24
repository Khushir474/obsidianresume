import os

from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import DirectoryReadTool
from obsidianresumeforge.tools.custom_tool import CachedFileReadTool, CachedFileWriterTool
from obsidianresumeforge.tools.judgeval_local_evaluator_runner import JudgevalLocalEvaluatorRunnerTool
from obsidianresumeforge.tools.cognee_memory_tool import CogneeMemoryTool





@CrewBase
class ObsidianresumeforgeCrew:
    """Obsidianresumeforge crew"""

    
    @agent
    def ats_keyword_extraction_specialist(self) -> Agent:
        
        
        return Agent(
            config=self.agents_config["ats_keyword_extraction_specialist"],
            
            
            tools=[				CachedFileReadTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            
            max_execution_time=None,
            llm=LLM(
                model="openrouter/google/gemini-2.5-flash-lite",
                
                
            ),
            
        )
        
    
    @agent
    def role_classification_judge(self) -> Agent:
        
        
        return Agent(
            config=self.agents_config["role_classification_judge"],
            
            
            tools=[				CachedFileReadTool(),
				CachedFileWriterTool(),
				DirectoryReadTool(),
				CogneeMemoryTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,


            max_execution_time=None,
            llm=LLM(
                model="openrouter/z-ai/glm-5.1",
                
                
            ),
            
        )
        
    
    @agent
    def expert_resume_writer_and_ats_optimizer(self) -> Agent:
        
        
        return Agent(
            config=self.agents_config["expert_resume_writer_and_ats_optimizer"],
            
            
            tools=[				CachedFileReadTool(),
				DirectoryReadTool(),
				CogneeMemoryTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,


            max_execution_time=None,
            llm=LLM(
                # model="openrouter/anthropic/claude-sonnet-4.6",
                model="openrouter/z-ai/glm-5.1",
                
                
            ),
            
        )
        
    
    @agent
    def pipeline_quality_evaluator(self) -> Agent:
        
        
        return Agent(
            config=self.agents_config["pipeline_quality_evaluator"],
            
            
            tools=[				CachedFileReadTool(),
				JudgevalLocalEvaluatorRunnerTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            
            max_execution_time=None,
            llm=LLM(
                model="openrouter/google/gemini-2.5-flash-lite",
                
                
            ),
            
        )
        
    
    @agent
    def pipeline_optimization_advisor(self) -> Agent:
        
        
        return Agent(
            config=self.agents_config["pipeline_optimization_advisor"],
            
            
            tools=[				CogneeMemoryTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            
            max_execution_time=None,
            llm=LLM(
                model="openrouter/anthropic/claude-haiku-4.5",
                
                
            ),
            
        )
        
    

    
    @task
    def classify_role(self) -> Task:
        return Task(
            config=self.tasks_config["classify_role"],
            markdown=False,
            
            
        )
    
    @task
    def extract_and_score_keywords(self) -> Task:
        return Task(
            config=self.tasks_config["extract_and_score_keywords"],
            markdown=False,
            
            
        )
    
    @task
    def write_tailored_latex_resume_and_export_pdf(self) -> Task:
        return Task(
            config=self.tasks_config["write_tailored_latex_resume_and_export_pdf"],
            markdown=False,
            
            
        )
    
    @task
    def evaluate_pipeline_output(self) -> Task:
        return Task(
            config=self.tasks_config["evaluate_pipeline_output"],
            markdown=False,
            
            
        )
    
    @task
    def log_run_and_generate_optimization_report(self) -> Task:
        return Task(
            config=self.tasks_config["log_run_and_generate_optimization_report"],
            markdown=False,
            
            
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the Obsidianresumeforge crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,

            chat_llm=LLM(model="openrouter/gpt-4o-mini"),
        )


