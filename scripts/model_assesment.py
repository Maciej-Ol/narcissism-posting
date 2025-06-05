from pathlib import Path
from typing import Union
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import re
from dotenv import load_dotenv

def model_assesment(
    file_to_open: Union[str, Path],
    target_name: Union[str, Path],
    model_used: str,
    model_role: str,
    score_type: str,
    score_level: str,
    column: str = "post",
) -> None:
    
    load_dotenv()

    file_to_open = Path(file_to_open)
    target_name = Path(target_name)

    model = ChatOpenAI(model=model_used, openai_api_key=os.getenv('OPENAI_API_KEY'), 
                        temperature=1,
                        timeout=15,
                        max_retries=3)

    data = pd.read_csv(file_to_open)
    testset = data[column].dropna().reset_index(drop=True)

    # output = pd.DataFrame(columns=['post', 'score', 'reason'])
    output = []
    errors = 0
    bad_responses = 0


    for i in range(testset.shape[0]):

        if i % 10 == 0:
            print(f"Processing {i} of {testset.shape[0]} Time: {pd.Timestamp.now()}")

        example_prompt = create_example_prompt(score_type=score_type)
        input = testset.iloc[i]
        final_prompt = create_final_prompt(example_prompt,model_role)
        
        # catch the errors
        try:
            response_str = get_response(final_prompt, model, input)
        except Exception as e:
            #print(f"Error: {e}")
            # Save the prompt that caused the error
            print(f"Error in prompt: {input}")
            errors += 1
            continue

        score, reason = get_score_and_reason(response_str, score_level)

        if score is None or reason is None:
            # Save the prompt that caused the error
            print(f"Problem in prompt: {input}")
            bad_responses += 1
        else: 
            row_to_add = {'post': input, 'score': score, 'reason': reason}
            output.append(row_to_add)

    print(f"Errors: {errors}")
    print(f"Bad responses: {bad_responses}")
    print(f"Output: {len(output)}")

    output_df = pd.DataFrame(output)
    output_df.to_csv(target_name, index = False)


# functions

def create_example_prompt(score_type):
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "Post"),
            ("ai", f"score: {score_type}, explanation: explanation"),
        ]        
    )
    return example_prompt

# create a final prompt
def create_final_prompt(example_prompt,model_role):
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("assistant", model_role),
            example_prompt,
            ("human", "{input}"),
        ]
    )
    return final_prompt

# Get the score and reason from the AI response
def get_score_and_reason(text, score_level):
    match_score = re.search(score_level, text, re.IGNORECASE)
    match_reason = re.search(r"(Explanation|Reason):?\*{0,2}:?(.*)", text,re.IGNORECASE | re.S)
    if match_score and match_reason:
        score = re.search(score_level, text, re.IGNORECASE).group(0)
        reason = re.search(r"(Explanation|Reason):?\*{0,2}:?(.*)", text,re.IGNORECASE | re.S).group(2)
    else:
        print(f"Wrong input: {text}")
        return None, None

    return score, reason

# get the response
def get_response(final_prompt, model, input):
    chain = final_prompt | model
    ai_message = chain.invoke({"input": input})
    response = ai_message.content
    return response

if __name__ == "__main__":
    model_assesment(
        file_to_open="data/processed/processed_without_ai.csv",
        target_name="data/reporting/emotional_affect/p_ea_abortion_o4mini_Maria.csv",
        model_used = "o4-mini",
        model_role = (
                # Role
                "You are a psychologist. "
                "Your name is Maria. You are a woman, 31 y.o., "
                "M.A. in Psychology, finished PhD school and submitted PhD dissertation. "
                "Your task is to assess to what extent positive emotional affect was involved "
                "in writing a text and provide an explanation for your assessment. "
                "Try to make your assessments more extreme. "
                "Use the following scale: "
                "Extremely high intensity of positive emotional affect, "
                "High intensity of positive emotional affect, "
                "Average intensity of positive emotional affect, " 
                "Low intensity of positive emotional affect, "
                "Extremely low intensity of positive emotional affect, "
                "The positive emotional affect is defined as: "
                "the Positive Affect (PA) dimension reflects one's level of pleasurable engagement "
                "with the environment. High PA is composed of terms reflecting enthusiasm, energy "
                "level, mental alertness, interest, joy, and deter- mination, whereas low PA is "
                "best defined by descriptors reflecting lethargy and lassitude (Clark, Watson, & Leeka, 1989)."

        ),
        score_type = "positive emotional affect score",
        score_level= "Extremely high intensity of positive emotional affect|High intensity of positive emotional affect|Average intensity of positive emotional affect|Low intensity of positive emotional affect|Extremely low intensity of positive emotional affect",
        column = "post_abortion",

        )



# OLD model roles:
#####
                # # Role
                # "You are a psychologist. "
                # "Your task is to assess to what extent varbal intelligence was involved in writing a text and provide an explanation for your assessment. "
                # "Try to make your assessments more extreme. "
                # # Scale
                # "Use the following scale: "
                # "Extremely high intelligence, "
                # "High intelligence, "
                # "Average intelligence, "
                # "Low intelligence, "
                # "Extremely low intelligence, "
                # # Definition of intelligence
                # "The intelligence is defined as: "
                # "Intelligence is a very general mental capability that, among other things, "
                # "involves the ability to reason, plan, solve problems, think abstractly, comprehend complex ideas, "
                # "learn quickly and learn from experience. It is not merely book learning, a narrow academic skill, "
                # "or test-taking smarts. Rather, it reflects a broader and deeper capability for comprehending our "
                # "surroundings - 'catching on', 'making sense' of things, or 'figuring out' what to do."
                # "The verbal intelligence is defined as: "
                # "The verbal intelligence quotient (VIQ) or verbal comprehension index (VCI) "
                # "represents the ability to access and apply acquired word knowledge, involving "
                # "verbal concept formation, reasoning and expression."
                # # Definition second part
                # # "Intelligence, so defined, can be measured, and intelligence tests measure it well. They are among "
                # # "the most accurate (in technical terms, reliable and valid) of all psychological tests and "
                # # "assessments. They do not measure creativity, character, personality, or other important "
                # # "differences among individuals, nor are they intended to."

                # score_type = "inteligence_score",

#####
#PA
#####
                ## Role
                # "You are a psychologist. "
                # "Your task is to assess to what extent positive emotional affect was involved "
                # "in writing a text and provide an explanation for your assessment. "
                # "Try to make your assessments more extreme. "
                # "Use the following scale: "
                # "Extremely high intensity of positive emotional affect, "
                # "High intensity of positive emotional affect, "
                # "Average intensity of positive emotional affect, " 
                # "Low intensity of positive emotional affect, "
                # "Extremely low intensity of positive emotional affect, "
                # "The positive emotional affect is defined as: "
                # "the Positive Affect (PA) dimension reflects one's level of pleasurable engagement "
                # "with the environment. High PA is composed of terms reflecting enthusiasm, energy "
                # "level, mental alertness, interest, joy, and deter- mination, whereas low PA is "
                # "best defined by descriptors reflecting lethargy and lassitude (Clark, Watson, & Leeka, 1989)."


                # score_type = "positive emotional affect score",
                # score_level= "Extremely high intensity of positive emotional affect|High intensity of positive emotional affect|Average intensity of positive emotional affect|Low intensity of positive emotional affect|Extremely low intensity of positive emotional affect"

#####
#NA
#####
                ## Role
                # "You are a psychologist. "
                # "Your task is to assess to what extent negative emotional affect was involved "
                # "in writing a text and provide an explanation for your assessment. "
                # "Try to make your assessments more extreme. "
                # "Use the following scale: "
                # "Extremely high intensity of negative emotional affect, "
                # "High intensity of negative emotional affect, "
                # "Average intensity of negative emotional affect, "
                # "Low intensity of negative emotional affect, "
                # "Extremely low intensity of negative emotional affect, "
                # "The negative emotional affect is defined as:"
                # "Negative Affect (NA) is a general factor of subjective distress, "
                # "and subsumes a wide range of negative emotional states, including fear, anger, guilt, "
                # "sadness, scorn, and disgust. Low NA is characterized by calm and relaxation "
                # "(Clark, Watson, & Leeka, 1989)."

                # score_type = "negative emotional affect score",
                # score_level = "Extremely high intensity of negative emotional affect|High intensity of negative emotional affect|Average intensity of negative emotional affect|Low intensity of negative emotional affect|Extremely low intensity of negative emotional affect"


# New model roles:
### Add:
                # "Your name is Maria. You are a woman, 31 y.o., "
                # "M.A. in Psychology, finished PhD school and submitted PhD dissertation. "

                # "Your name is Ania. You are a woman,"
                # " 28 y.o., M.A. in Psychology, On the 2nd year in the PhD school."


                # "Your name is Marysia. You are a woman, 23 y.o., "
                # "5th (last) year psychology student."

