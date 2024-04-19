import os
import sys
from openai import OpenAI
client = OpenAI(api_key=os.environ['OPENAI_KEY'])
openai_models=["gpt-3.5-turbo-instruct",
                "davinci-002",
                "text-embedding-3-large", # 못쓴다함
                "babbage-002",
                "gpt-4"]
openai_model=openai_models[0]


def playing_with_openai(instr):
    question = f"""
    I: my advanced robotics score is 100. A: GOOD JOB
    I: my advanced robotics score is 80. A: GOOD JOB
    I: my advanced robotics score is 60. A: GOOD JOB
    I: my advanced robotics score is 40. A: YOU ARE BAD STUDENT
    I: {instr}. A:"""
    response = client.completions.create(model=openai_model, prompt=question, max_tokens=64, temperature=0.0)
    #print(response)
    result=response.choices[0].text.strip()
    print("result: ", result)

def ha(instr):
    question = f"""
    I: {instr}. A:"""
    response = client.completions.create(model=openai_model, prompt=question, max_tokens=64, temperature=0.0)
    #print(response)
    result=response.choices[0].text.strip()
    print("result: ", result)


if __name__ == "__main__":
    input_str = sys.argv[1]
    ha(input_str)
    #playing_with_openai(input_str)