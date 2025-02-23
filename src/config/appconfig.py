from dotenv import load_dotenv
import os
load_dotenv()


groq_key = os.getenv("GROQ_API_KEY")
hf_key = os.getenv("HF_KEY")

# self.system_logger.info("QA chain created successfully")
#             self.userops_logger.info(
#                 f"""
#                 User Request:
#                 -----log response-----
#                 User data: {question}
#                 """
#             )
#             response =  qa_chain.invoke(question)
#             self.llm_logger.info(
#                 f"""
#                 User Request:
#                 -----log response-----
#                 User data: {response}
#                 """
#             )