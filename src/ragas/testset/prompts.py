from langchain_core.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt

reasoning_question_prompt = Prompt(
    name="reasoning_question",
    instruction="""Complicate the given question by rewriting question into a multi-hop reasoning question based on the provided context.
    Answering the question should require the reader to make multiple logical connections or inferences using the information available in given context.
    Rules to follow when rewriting question:
    1. Ensure that the rewritten question can be answered entirely from the information present in the contexts.
    2. Do not frame questions that contains more than 15 words. Use abbreviation wherever possible.
    3. Make sure the question is clear and unambiguous.
    4. phrases like 'based on the provided context','according to the context',etc are not allowed to appear in the question.""",
    examples=[
        {
            "question": "What is the capital of France?",
            "context": "France is a country in Western Europe. It has several cities, including Paris, Lyon, and Marseille. Paris is not only known for its cultural landmarks like the Eiffel Tower and the Louvre Museum but also as the administrative center.",
            "output": "Linking the Eiffel Tower and administrative center, which city stands as both?",
        },
        {
            "question": "What does the append() method do in Python?",
            "context": "In Python, lists are used to store multiple items in a single variable. Lists are one of 4 built-in data types used to store collections of data. The append() method adds a single item to the end of a list.",
            "output": "If a list represents a variable collection, what method extends it by one item?",
        },
    ],
    input_keys=["question", "context"],
    output_key="output",
    output_type="str",
    language="english",
)


multi_context_question_prompt = Prompt(
    name="multi_context_question",
    instruction="""
    The task is to rewrite and complicate the given question in a way that answering it requires information derived from both context1 and context2. 
    Follow the rules given below while rewriting the question.
        1. The rewritten question should not be very long. Use abbreviation wherever possible.
        2. The rewritten question must be reasonable and must be understood and responded by humans.
        3. The rewritten question must be fully answerable from information present in context1 and context2. 
        4. Read and understand both contexts and rewrite the question so that answering requires insight from both context1 and context2.
        5. phrases like 'based on the provided context','according to the context?',etc are not allowed to appear in the question.""",
    examples=[
        {
            "question": "What process turns plants green?",
            "context1": "Chlorophyll is the pigment that gives plants their green color and helps them photosynthesize.",
            "context2": "Photosynthesis in plants typically occurs in the leaves where chloroplasts are concentrated.",
            "output": "In which plant structures does the pigment responsible for their verdancy facilitate energy production?",
        },
        {
            "question": "How do you calculate the area of a rectangle?",
            "context1": "The area of a shape is calculated based on the shape's dimensions. For rectangles, this involves multiplying the length and width.",
            "context2": "Rectangles have four sides with opposite sides being equal in length. They are a type of quadrilateral.",
            "output": "What multiplication involving equal opposites yields a quadrilateral's area?",
        },
    ],
    input_keys=["question", "context1", "context2"],
    output_key="output",
    output_type="str",
    language="english",
)

conditional_question_prompt = Prompt(
    name="conditional_question",
    instruction="""Rewrite the provided question to increase its complexity by introducing a conditional element.
    The goal is to make the question more intricate by incorporating a scenario or condition that affects the context of the question.
    Follow the rules given below while rewriting the question.
        1. The rewritten question should not be longer than 25 words. Use abbreviation wherever possible.
        2. The rewritten question must be reasonable and must be understood and responded by humans.
        3. The rewritten question must be fully answerable from information present context.
        4. phrases like 'provided context','according to the context?',etc are not allowed to appear in the question.""",
    examples=[
        {
            "question": "What is the function of the roots of a plant?",
            "context": "The roots of a plant absorb water and nutrients from the soil, anchor the plant in the ground, and store food.",
            "output": "What dual purpose do plant roots serve concerning soil nutrients and stability?",
        },
        {
            "question": "How do vaccines protect against diseases?",
            "context": "Vaccines protect against diseases by stimulating the body's immune response to produce antibodies, which recognize and combat pathogens.",
            "output": "How do vaccines utilize the body's immune system to defend against pathogens?",
        },
    ],
    input_keys=["question", "context"],
    output_key="output",
    output_type="str",
    language="english",
)


compress_question_prompt = Prompt(
    name="compress_question",
    instruction="""Rewrite the following question.\n- make it more indirect and shorter\n- keep it clear, reasonable and easy to understand\n- retain full essence of the original question\n- the same answer should satisfy both old and new question.\n
The goal is to create a question that conveys the same meaning but in a less direct manner. The rewritten question should be concise so use shorter forms and reasonable common abbreviations wherever possible.
Respond with the question content only without any extra comments.""",
    examples=[
        {
            "question": "What is the distance between the Earth and the Moon?",
            "output": "How far is the Moon from Earth?",
        },
        {
            "question": "What ingredients are required to bake a chocolate cake?",
            "output": "What's needed for a chocolate cake?",
        },
    ],
    input_keys=["question"],
    output_key="output",
    output_type="str",
    language="english",
)


conversational_question_prompt = Prompt(
    name="conversation_question",
    instruction="""Reformat the provided question into two separate questions as if it were to be part of a conversation. Each question should focus on a specific aspect or subtopic related to the original question.
    Follow the rules given below while rewriting the question.
        1. The rewritten question should not be longer than 25 words. Use abbreviation wherever possible.
        2. The rewritten question must be reasonable and must be understood and responded by humans.
        3. The rewritten question must be fully answerable from information present context.
        4. phrases like 'provided context','according to the context?',etc are not allowed to appear in the question.""",
    examples=[
        {
            "question": "What are the advantages and disadvantages of remote work?",
            "output": {
                "first_question": "What are the benefits of remote work?",
                "second_question": "On the flip side, what challenges are encountered when working remotely?",
            },
        }
    ],
    input_keys=["question"],
    output_key="output",
    output_type="json",
    language="english",
)


question_answer_prompt = Prompt(
    name="answer_formulate",
    instruction="""Act as an expert in the domain of the given "INPUT" context. Using only the provided context, answer the question to the best of your ability. Assume the context should contain the necessary information to answer the question. To find the answer:

1. Carefully read the entire context and identify the parts most relevant to answering the question.
2. Break the question down into key components and look for related details in the context that address each component.
3. Consider the overall themes and topics covered in the context to help interpret the question and fill in any gaps.
4. Combine details from different parts of the context as needed to arrive at a complete answer, even if not explicitly stated. Make reasonable inferences based on the available information.
5. A verdict of '1' can be given for a closely related, concise comprehensive answer that addresses enough of the key aspects of the question, even if no direct strightforward answer is possible due to some minority of the details are missing. Carefully weigh whether the answer is constructive and informative enough to justify a '1' verdict, or if it emphasizes too many unknowns, making a '-1' verdict more appropriate.""",
    examples=[
        {
            "context": """Climate change is significantly influenced by human activities, notably the emission of greenhouse gases from burning fossil fuels. The increased greenhouse gas concentration in the atmosphere traps more heat, leading to global warming and changes in weather patterns.""",
            "question": "How do human activities contribute to climate change?",
            "answer": {
                "answer": "Human activities contribute to climate change primarily through the emission of greenhouse gases from burning fossil fuels. These emissions increase the concentration of greenhouse gases in the atmosphere, which traps more heat and leads to global warming and altered weather patterns.",
                "verdict": "1",
            },
        },
        {
            "context": """The concept of artificial intelligence (AI) has evolved over time, but it fundamentally refers to machines designed to mimic human cognitive functions. AI can learn, reason, perceive, and, in some instances, react like humans, making it pivotal in fields ranging from healthcare to autonomous vehicles.""",
            "question": "What are the key capabilities of artificial intelligence?",
            "answer": {
                "answer": "Artificial intelligence is designed to mimic human cognitive functions, with key capabilities including learning, reasoning, perception, and reacting to the environment in a manner similar to humans. These capabilities make AI pivotal in various fields, including healthcare and autonomous driving.",
                "verdict": "1",
            },
        },
        {
            "context": """The novel "Pride and Prejudice" by Jane Austen revolves around the character Elizabeth Bennet and her family. The story is set in the 19th century in rural England and deals with issues of marriage, morality, and misconceptions.""",
            "question": "What year was 'Pride and Prejudice' published?",
            "answer": {
                "answer": "The answer to given question is not present in context",
                "verdict": "-1",
            },
        },
    ],
    input_keys=["context", "question"],
    output_key="answer",
    output_type="json",
    language="english",
)

keyphrase_extraction_prompt = Prompt(
    name="keyphrase_extraction",
    instruction="Extract 3 to 5 unique, important keyphrases: \n- summarize the main topics of 'Your actual INPUT' text content.\n- Provide the output as a JSON object with a 'keyphrases' key \n- list minimum 3 and maximum 5 keyphrases.\n",
#    instruction="Extract the concise and short non-empty 3 to 5 elements list of maximum 5 items - unique, distinct, not related, non repetitive, general - most important, general keyphrases summarising and characterizing main topics of the whole content of 'Your actual INPUT' and ensure to ignore and not refer to keyphrases from examples. Provide just single output response strictly in valid JSON format that follows examples structure.",
    examples=[
        {
            "text": "A black hole is a region of spacetime where gravity is so strong that nothing, including light and other electromagnetic waves, has enough energy to escape it. The theory of general relativity predicts that a sufficiently compact mass can deform spacetime to form a black hole.",
            "json_output": {
                "keyphrases": [
                    "Black hole",
                    "Region of spacetime",
                    "Strong gravity",
                    "Light and electromagnetic waves",
                    "Theory of general relativity",
                ]
            },
        },
        {
            "text": "The Great Wall of China is an ancient series of walls and fortifications located in northern China, built around 500 years ago. This immense wall stretches over 13,000 miles and is a testament to the skill and persistence of ancient Chinese engineers.",
            "json_output": {
                "keyphrases": [
                    "Great Wall of China",
                    "Ancient fortifications",
                    "Northern China",
                ]
            },
        },
        {
            "text": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Locally nicknamed 'La dame de fer', it was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair and was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world.",
            "json_output": {
                "keyphrases": [
                    "Eiffel Tower",
                    "Wrought-iron lattice tower",
                    "Paris, France",
                    "Gustave Eiffel",
                    "Cultural icon of France",
                ]
            },
        },
    ],
    input_keys=["text"],
    output_key="json_output",
    output_type="json",
)


seed_question_prompt = Prompt(
    name="seed_question",
    instruction="Generate a question that can be fully answered only by properly referring various details contained in the given context and so that all the details required to answer exist in the context. The question should be formed using specific topic and be possible to find all the properly related details in the context.",
    examples=[
        {
            "context": "Photosynthesis in plants involves converting light energy into chemical energy, using chlorophyll and other pigments to absorb light. This process is crucial for plant growth and the production of oxygen.",
            "keyphrase": "Photosynthesis",
            "question": "What is the role of photosynthesis in plant growth?",
        },
        {
            "context": "The Industrial Revolution, starting in the 18th century, marked a major turning point in history as it led to the development of factories and urbanization.",
            "keyphrase": "Industrial Revolution",
            "question": "How did the Industrial Revolution mark a major turning point in history?",
        },
        {
            "context": "The process of evaporation plays a crucial role in the water cycle, converting water from liquid to vapor and allowing it to rise into the atmosphere.",
            "keyphrase": "Evaporation",
            "question": "Why is evaporation important in the water cycle?",
        },
    ],
    input_keys=["context", "keyphrase"],
    output_key="question",
    output_type="str",
)

main_topic_extraction_prompt = Prompt(
    name="main_topic_extraction",
    instruction="Identify and extract the two main topics discussed in depth in the given text.",
    examples=[
        {
            "text": "Blockchain technology presents a decentralized ledger that ensures the integrity and transparency of data transactions. It underpins cryptocurrencies like Bitcoin, providing a secure and immutable record of all transactions. Beyond finance, blockchain has potential applications in supply chain management, where it can streamline operations, enhance traceability, and improve fraud prevention. It allows for real-time tracking of goods and transparent sharing of data among participants.",
            "output": {
                "topics": [
                    "Blockchain technology and its foundational role in cryptocurrencies",
                    "Applications of blockchain in supply chain management",
                ]
            },
        },
        {
            "text": "Telemedicine has revolutionized the way healthcare is delivered, particularly in rural and underserved areas. It allows patients to consult with doctors via video conferencing, improving access to care and reducing the need for travel. Another significant advancement in healthcare is precision medicine, which tailors treatments to individual genetic profiles. This approach has led to more effective therapies for a variety of conditions, including certain cancers and chronic diseases.",
            "output": {
                "topics": [
                    "Telemedicine and its impact on healthcare accessibility",
                    "Precision medicine and its role in tailoring treatments to genetic profiles",
                ]
            },
        },
    ],
    input_keys=["text"],
    output_key="output",
    output_type="json",
)


find_relevant_context_prompt = Prompt(
    name="find_relevant_context",
    instruction="Given a question and set of contexts, find the most relevant contexts from the set to answer the question. Provide at least one context id and ensure all are matching provided set.",
    examples=[
        {
            "question": "What is the capital of France?",
            "contexts": {
                "1": "France is a country in Western Europe. It has several cities, including Paris, Lyon, and Marseille. Paris is not only known for its cultural landmarks like the Eiffel Tower and the Louvre Museum but also as the administrative center.",
                "2": "The capital of France is Paris. It is also the most populous city in France, with a population of over 2 million people. Paris is known for its cultural landmarks like the Eiffel Tower and the Louvre Museum.",
                "3": "Paris is the capital of France. It is also the most populous city in France, with a population of over 2 million people. Paris is known for its cultural landmarks like the Eiffel Tower and the Louvre Museum.",
            },
            "output": {
                "relevant_contexts": [1, 2],
            },
        },
        {
            "question": "How does caffeine affect the body and what are its common sources?",
            "contexts": {
                "1": "Regular physical activity is essential for maintaining good health. It can help control weight, combat health conditions, boost energy, and promote better sleep.",
                "2": "Common sources of caffeine include coffee, tea, cola, and energy drinks. These beverages are consumed worldwide and are known for providing a quick boost of energy.",
                "3": "Caffeine is a central nervous system stimulant. It can temporarily ward off drowsiness and restore alertness. It primarily affects the brain, where it alters the function of neurotransmitters.",
            },
            "output": {"relevant_contexts": [2, 3]},
        },
    ],
    input_keys=["question", "contexts"],
    output_key="output",
    output_type="json",
    language="english",
)


question_rewrite_prompt = Prompt(
    name="rewrite_question",
    instruction="""Given a context, question and feedback, reword the question preserving all the details and make it easier to understand, focus on improving its clarity and answerability based on the feedback provided. Make sure the new question is straightforward and easy to understand.""",
    examples=[
        {
            "context": "The Eiffel Tower was constructed using iron and was originally intended as a temporary exhibit for the 1889 World's Fair held in Paris. Despite its initial temporary purpose, the Eiffel Tower quickly became a symbol of Parisian ingenuity and an iconic landmark of the city, attracting millions of visitors each year. The tower's design, created by Gustave Eiffel, was initially met with criticism from some French artists and intellectuals, but it has since been celebrated as a masterpiece of structural engineering and architectural design.",
            "question": "Who created the design for the Tower?",
            "feedback": "The question asks about the creator of the design for 'the Tower', but it does not specify which tower it refers to. There are many towers worldwide, and without specifying the exact tower, the question is unclear and unanswerable. To improve the question, it should include the name or a clear description of the specific tower in question.",
            "output": "Who created the design for the Eiffel Tower?",
        },
        {
            "context": "'Exploring Zero-Shot Learning in Neural Networks' was published by Smith and Lee in 2021, focusing on the application of zero-shot learning techniques in artificial intelligence.",
            "question": "What datasets were used for the zero-shot evaluations in this study?",
            "feedback": "The question asks about the datasets used for zero-shot evaluations in 'this study', without specifying or providing any details about the study in question. This makes the question unclear for those who do not have access to or knowledge of the specific study. To improve clarity and answerability, the question should specify the study it refers to, or provide enough context about the study for the question to be understood and answered independently.",
            "output": "What datasets were used for the zero-shot evaluations Exploring Zero-Shot Learning in Neural Networks paper?",
        },
    ],
    input_keys=["context", "question", "feedback"],
    output_key="output",
    output_type="str",
    language="english",
)

### Filters


class ContextScoring(BaseModel):
    clarity: int
    depth: int
    structure: int
    relevance: int


class QuestionFilter(BaseModel):
    feedback: str
    verdict: int


class EvolutionElimination(BaseModel):
    reason: str
    verdict: int


context_scoring_parser = RagasoutputParser(pydantic_object=ContextScoring)
question_filter_parser = RagasoutputParser(pydantic_object=QuestionFilter)
evolution_elimination_parser = RagasoutputParser(pydantic_object=EvolutionElimination)

context_scoring_prompt = Prompt(
    name="score_context",
    instruction="""
    Given a context, perform the following task and output the answer in VALID JSON format: Assess the provided context and assign a numerical score of 1 (Low), 2 (Medium), or 3 (High) for each of the following criteria in your JSON response:

clarity: Evaluate the precision and understandability of the information presented. High scores (3) are reserved for contexts that are both precise in their information and easy to understand. Low scores (1) are for contexts where the information is vague or hard to comprehend.
depth: Determine the level of detailed examination and the inclusion of innovative insights within the context. A high score indicates a comprehensive and insightful analysis, while a low score suggests a superficial treatment of the topic.
structure: Assess how well the content is organized and whether it flows logically. High scores are awarded to contexts that demonstrate coherent organization and logical progression, whereas low scores indicate a lack of structure or clarity in progression.
relevance: Judge the pertinence of the content to the main topic, awarding high scores to contexts tightly focused on the subject without unnecessary digressions, and low scores to those that are cluttered with irrelevant information.
Structure your JSON output to reflect these criteria as keys with their corresponding scores as values
    """,
    output_format_instruction=get_json_format_instructions(ContextScoring),
    examples=[
        {
            "context": "The Pythagorean theorem is a fundamental principle in geometry. It states that in a right-angled triangle, the square of the length of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the lengths of the other two sides. This can be written as a^2 + b^2 = c^2 where c represents the length of the hypotenuse, and a and b represent the lengths of the other two sides.",
            "output": ContextScoring.parse_obj(
                {"clarity": 3, "depth": 1, "structure": 3, "relevance": 3}
            ).dict(),
        },
        {
            "context": "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time.",
            "output": ContextScoring.parse_obj(
                {"clarity": 3, "depth": 2, "structure": 3, "relevance": 3}
            ).dict(),
        },
        {
            "context": "I love chocolate. It's really tasty. Oh, and by the way, the earth orbits the sun, not the other way around. Also, my favorite color is blue.",
            "output": ContextScoring.parse_obj(
                {"clarity": 2, "depth": 1, "structure": 1, "relevance": 1}
            ).dict(),
        },
    ],
    input_keys=["context"],
    output_key="output",
    output_type="json",
    language="english",
)


filter_question_prompt = Prompt(
    name="filter_question",
    instruction="""
Asses the given question for clarity and answerability given enough domain knowledge, consider the following criteria:
1.Independence: Can the question be understood and answered without needing additional context or access to external references not provided within the question itself? Questions should be self-contained, meaning they do not rely on specific documents, tables, or prior knowledge not shared within the question.
2.Clear Intent: Is it clear what type of answer or information the question seeks? The question should convey its purpose without ambiguity, allowing for a direct and relevant response.
Based on these criteria, assign a verdict of "1" if a question is specific, independent, and has a clear intent, making it understandable and answerable based on the details provided. Assign "0" if it fails to meet one or more of these criteria due to vagueness, reliance on external references, or ambiguity in intent.
Provide feedback and a verdict in JSON format, including suggestions for improvement if the question is deemed unclear. Highlight aspects of the question that contribute to its clarity or lack thereof, and offer advice on how it could be reframed or detailed for better understanding and answerability.
""",
    output_format_instruction=get_json_format_instructions(QuestionFilter),
    examples=[
        {
            "question": "What is the discovery about space?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "The question is too vague and broad, asking for a 'discovery about space' without specifying any particular aspect, time frame, or context of interest. This could refer to a wide range of topics, from the discovery of new celestial bodies to advancements in space travel technology. To improve clarity and answerability, the question could specify the type of discovery (e.g., astronomical, technological), the time frame (e.g., recent, historical), or the context (e.g., within a specific research study or space mission).",
                    "verdict": "0",
                }
            ).dict(),
        },
        {
            "question": "How does ALMA-13B-R perform compared to other translation models in the WMT'23 study, based on the results in context1 and context2?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "This question asks for a comparison of the ALMA-13B-R model's performance against other translation models within the WMT'23 study, specifically referring to results in 'context1' and 'context2'. While it clearly specifies the model of interest (ALMA-13B-R) and the study (WMT'23), it assumes access to and understanding of 'context1' and 'context2' without explaining what these contexts entail. This makes the question unclear for those not familiar with the WMT'23 study or these specific contexts. To improve clarity and answerability for a broader audience, the question could benefit from defining or describing 'context1' and 'context2' or explaining the criteria used for comparison in these contexts.",
                    "verdict": "0",
                }
            ).dict(),
        },
        {
            "question": "How do KIWI-XXL and XCOMET compare to the gold standard references in Table 1 in terms of evaluation scores, translation model performance, and success rate in surpassing the references?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "The question requests a comparison between KIWI-XXL and XCOMET models and gold standard references in 'Table 1', focusing on evaluation scores, translation model performance, and success rates in surpassing the references. It specifies the models and criteria for comparison, making the intent clear. However, the question assumes access to 'Table 1' without providing its content or context, making it unclear for those without direct access to the source material. To be clearer and more answerable for a general audience, the question could include a brief description of the content or key findings of 'Table 1', or alternatively, frame the question in a way that does not rely on specific, unpublished documents.",
                    "verdict": 0,
                }
            ).dict(),
        },
        {
            "question": "What is the configuration of UL2 training objective in OpenMoE and why is it a better choice for pre-training?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "The question asks for the configuration of the UL2 training objective within the OpenMoE framework and the rationale behind its suitability for pre-training. It is clear in specifying the topic of interest (UL2 training objective, OpenMoE) and seeks detailed information on both the configuration and the reasons for its effectiveness in pre-training. However, the question might be challenging for those unfamiliar with the specific terminology or the context of OpenMoE and UL2. For broader clarity and answerability, it would be helpful if the question included a brief explanation or context about OpenMoE and the UL2 training objective, or clarified the aspects of pre-training effectiveness it refers to (e.g., efficiency, accuracy, generalization).",
                    "verdict": 1,
                }
            ).dict(),
        },
        {
            "question": "What is the detailed configuration of the UL2 training objective in OpenMoE, based on the provided context?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "The question seeks detailed information on the UL2 training objective's configuration within the OpenMoE framework, mentioning 'the provided context' without actually including or describing this context within the query. This makes the question unclear for those who do not have access to the unspecified context. For the question to be clear and answerable, it needs to either include the relevant context directly within the question or be framed in a way that does not require external information. Detailing the specific aspects of the configuration of interest (e.g., loss functions, data augmentation techniques) could also help clarify the query.",
                    "verdict": 0,
                }
            ).dict(),
        },
    ],
    input_keys=["question"],
    output_key="output",
    output_type="json",
    language="english",
)

evolution_elimination_prompt = Prompt(
    name="evolution_elimination",
    instruction="""Check if the given two questions are equal based on following requirements:
    1. They have same constraints and requirements.
    2. They have same depth and breadth of the inquiry.
    3. They are mostly equivalent and so a correct answer should satisfy both questions. 
    Output verdict as 1 if they are equal and 0 if they are not""",
    output_format_instruction=get_json_format_instructions(EvolutionElimination),
    examples=[
        {
            "question1": "What are the primary causes of climate change?",
            "question2": "What factors contribute to global warming?",
            "output": EvolutionElimination.parse_obj(
                {
                    "reason": "While both questions deal with environmental issues, 'climate change' encompasses broader changes than 'global warming', leading to different depths of inquiry.",
                    "verdict": 0,
                }
            ).dict(),
        },
        {
            "question1": "How does photosynthesis work in plants?",
            "question2": "Can you explain the process of photosynthesis in plants?",
            "output": EvolutionElimination.parse_obj(
                {
                    "reason": "Both questions ask for an explanation of the photosynthesis process in plants, sharing the same depth, breadth, and requirements for the answer.",
                    "verdict": 1,
                }
            ).dict(),
        },
        {
            "question1": "What are the health benefits of regular exercise?",
            "question2": "Can you list the advantages of exercising regularly for health?",
            "output": EvolutionElimination.parse_obj(
                {
                    "reason": "Both questions seek information about the positive effects of regular exercise on health. They require a similar level of detail in listing the health benefits.",
                    "verdict": 1,
                }
            ).dict(),
        },
    ],
    input_keys=["question1", "question2"],
    output_key="output",
    output_type="json",
    language="english",
)
