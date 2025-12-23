from secret_key import second_key
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain



os.environ['OPENAI_API_KEY'] = second_key


llm = OpenAI(temperature=0.6)

def generate_restaurant_idea_and_items(cuisine):

    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template = " I want to open a restaurant for {cuisine} food. Suggest a fency name food items"

    )

    name_chain  = LLMChain(llm=llm, prompt=prompt_template_name, output_key = 'restaurant_name')

    prompt_template_items = PromptTemplate(
    input_variables= ['restaurant_name'],
    template= """ Suggest some menu items for {restaurant_name}."""

    )
    food_items_chain = LLMChain(llm=llm, prompt = prompt_template_items, output_key = 'menu_items')
    
    Chain = SequentialChain(
        chains = [name_chain, food_items_chain], 
        input_variables = ['cuisine'],
        output_variables = ['restaurant_name', 'menu_items']
    )
    response = Chain({'cuisine': cuisine})

    return response


if __name__ == "__main__":
    print(generate_restaurant_idea_and_items("Thai"))