import logging
from typing import List, Dict, Any
import dspy
import asyncio
from metrics import comprehensive_metric, is_answer_fully_correct, factuality_metric
from utils.utils import check_answer_length

logger = logging.getLogger(__name__)

class QuestionAnswerSignature(dspy.Signature):
    input: str = dspy.InputField(
        desc=(
            "The combined input containing both the question and context. "
            "The format should be 'question: <question_text> context: <context_text>'."
        )
    )
    answer: str = dspy.OutputField(
        desc=(
            "The generated answer to the question. The answer should be concise, directly address "
            "the question, and be grounded in the provided context to ensure factual accuracy."
        )
    )

    def forward(self, input: str, max_tokens: int = 8192) -> Dict[str, str]:
        try:
            # Parse the input to extract question and context
            parts = input.split(' context: ', 1)
            question = parts[0].replace('question: ', '').strip()
            context = parts[1].strip() if len(parts) > 1 else ""

            logger.debug(f"Generating answer for question: '{question}' with context length: {len(context)} characters.")
            answer = self.language_model.generate(
                prompt=(
                    f"You are an expert in qualitative research and thematic analysis.\n\n"
                    f"**Guidelines**:\n"
                    f"- **Relevance:** Extract quotations that are closely related to the key themes.\n"
                    f"- **Diversity:** Ensure a range of perspectives and viewpoints.\n"
                    f"- **Clarity:** Choose clear and understandable quotations.\n"
                    f"- **Impact:** Select impactful quotations that highlight significant aspects of the data.\n"
                    f"- **Authenticity:** Maintain original expressions from participants.\n\n"
                    f"**Transcript Chunk**:\n{question}\n\n"
                    f"**Context:**\n{context}\n\n"
                    f"**Task:** Extract **3-5** relevant quotations from the transcript chunk based on the context provided. "
                    f"Provide each quotation in the following JSON format within a list:\n\n"
                    f"```json\n"
                    f"[\n"
                    f"    {{\"QUOTE\": \"This is the first quotation.\"}},\n"
                    f"    {{\"QUOTE\": \"This is the second quotation.\"}},\n"
                    f"    {{\"QUOTE\": \"This is the third quotation.\"}}\n"
                    f"]\n"
                    f"```"
                    f"Ensure that the response is a valid JSON array containing all relevant quotations. "
                    f"If no quotations are available, respond with an empty array `[]`."
                ),
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                n=1,
                stop=None
            ).strip()
            logger.info(f"Generated answer for question: '{question}'")
            logger.debug(f"Answer length: {len(answer)} characters.")
            return {"answer": answer}
        except Exception as e:
            logger.error(f"Error in QuestionAnswerSignature.forward: {e}", exc_info=True)
            return {"answer": "I'm sorry, I couldn't generate an answer at this time."}

try:
    qa_module = dspy.Program.load("optimized_program.json")
    logger.info("Optimized DSPy program loaded successfully.")
except Exception as e:
    try:
        qa_module = dspy.TypedChainOfThought(QuestionAnswerSignature)
        logger.info("Unoptimized DSPy module initialized successfully.")
    except Exception as inner_e:
        logger.error(f"Error initializing unoptimized DSPy module: {inner_e}", exc_info=True)
        raise

async def generate_answer(input: str, max_tokens: int = 8192) -> str:
    try:
        logger.debug(f"Generating answer for input with length: {len(input)} characters.")
        answer = await asyncio.to_thread(qa_module, input=input, max_tokens=max_tokens)
        return answer.get("answer", "I'm sorry, I couldn't generate an answer at this time.")
    except Exception as e:
        logger.error(f"Error in generate_answer: {e}", exc_info=True)
        return "I'm sorry, I couldn't generate an answer at this time."

async def evaluate_answer(example: Dict[str, Any], pred: Dict[str, Any]) -> bool:
    try:
        logger.debug(f"Evaluating answer for input: '{example.get('input', '')}'")
        return await asyncio.to_thread(is_answer_fully_correct, example, pred)
    except Exception as e:
        logger.error(f"Error in evaluate_answer: {e}", exc_info=True)
        return False

async def generate_answer_dspy(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger.debug(f"Entering generate_answer_dspy with query='{query}' and {len(retrieved_chunks)} retrieved_chunks.")
    try:
        context = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            chunk_content = chunk['chunk'].get('original_content', '')
            chunk_context = chunk['chunk'].get('contextualized_content', '')
            context += f"Chunk {i}:\n"
            context += f"Content: {chunk_content}\n"
            context += f"Context: {chunk_context}\n\n"

        if not context.strip():
            logger.warning(f"No valid context found for query '{query}'.")
            return {
                "answer": "I'm sorry, I couldn't find relevant information to answer your question.",
                "used_chunks": [],
                "num_chunks_used": 0
            }

        logger.debug(f"Formatted context with {len(retrieved_chunks)} sequential chunks:\n{context[:200]}...")
        used_chunks_info = [
            {
                "chunk_id": chunk['chunk'].get('chunk_id', ''),
                "doc_id": chunk['chunk'].get('doc_id', ''),
                "content_snippet": chunk['chunk'].get('original_content', '')[:100] + "..."
            }
            for chunk in retrieved_chunks
        ]
        logger.info(f"Total number of chunks used for context in query '{query}': {len(used_chunks_info)}")
        logger.info(f"Chunks used for context: {used_chunks_info}")
        input_data = f"question: {query} context: {context}"
        answer = await generate_answer(input_data)

        if not answer:
            logger.warning(f"No answer generated for query '{query}'.")
            return {
                "answer": "I'm sorry, I couldn't generate an answer at this time.",
                "used_chunks": used_chunks_info,
                "num_chunks_used": len(used_chunks_info)
            }

        logger.debug(f"Generated answer for query '{query}': {answer}")
        logger.info(f"Number of chunks used for query '{query}': {len(used_chunks_info)}")
        example = {
            "context": context,
            "question": query
        }
        pred = {
            "answer": answer
        }
        suggestion = await evaluate_answer(example, pred)
        return {
            "answer": answer,
            "used_chunks": used_chunks_info,
            "num_chunks_used": len(used_chunks_info)
        }
    except Exception as e:
        logger.error(f"Error generating answer via DSPy for query '{query}': {e}", exc_info=True)
        return {
            "answer": "I'm sorry, I couldn't generate an answer at this time.",
            "used_chunks": [],
            "num_chunks_used": 0
        }

async def generate_answers_dspy(queries: List[str], retrieved_chunks_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    tasks = [
        generate_answer_dspy(query, retrieved_chunks)
        for query, retrieved_chunks in zip(queries, retrieved_chunks_list)
    ]
    return await asyncio.gather(*tasks)

def is_answer_factually_correct(example: Dict[str, Any], pred: Dict[str, Any]) -> bool:
    score = factuality_metric(example, pred)
    logger.debug(f"Factuality score: {score}")
    return score == 1
