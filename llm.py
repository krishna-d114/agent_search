from openai import OpenAI
import os

def rank(titles_and_urls,query):
    client = OpenAI(
        base_url ="https://openrouter.ai/api/v1",
        api_key =os.environ.get("OPENROUTER_API_KEY"),
    )
    system_prompt = """
    you are a very intelligent indivdual, given a query and some search_results, you need to tell me which of the following urls 
    most likely going to answer the given query
    return just the title.
    """
    completion = client.chat.completions.create(
        model = "openrouter/owl-alpha",
        messages = [
            {"role":"system","content":system_prompt},
            {"role":"user","content":f"here is the query:{query} and here is the bunch of results = {titles_and_urls}.\n give the page that will most likely answer the query."}
        ]
    )
    content = completion.choices[0].message.content
    return content

def niche(query):
    client = OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key =os.environ.get("OPENROUTER_API_KEY"),
    )
    system_prompt = """
     given the query, you need to tell if the required information is niche or not niche.
     Niche information just means it needs more searching or more results have to be analyzed.
     Your job is to make sure that response is as accurate as possible. just remember if you dont know it, its probably niche.
     just return true or false, nothing else.
     return true if its a niche query
     return false if its not.
     and also provide a one line justification.
     so required format is:
     ""true/false"+{reason}"
    """

    completions = client.chat.completions.create(
        model = "openrouter/owl-alpha",
        messages = [
            {"role":"system","content":system_prompt},
            {"role":"user","content":f"here's a query:{query},\n you just have to classify whether the query is niche or not niche and also provide one line justififcation"},            
        ]
    )
    content = completions.choices[0].message.content
    return content


def query_decomposition(query,niche):
#the goal of this function is to basically create alternate queries to answer the actual query.
    client = OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key = os.environ.get("OPENROUTER_API_KEY")
    )
    system_prompt = """
    you are basically an expert in reasoning.
    Given a query, that cannot be answered by a chatbot, you need to generate alternate queries which
    will be searched in the browser.
    your inputs are:
    1.query
    2.nicheness(you will be provided a simple string saying "niche" or "not niche")
    the number of alternate queries to be generated are 10, if its "niche"
    the number of alternate queries to be generated are 5, if its "not niche"
    return a clean string containing:
    "
    1. alternate_query-1
    2. alternate_query_2
    .
    .
    .
    "
    """

    completions = client.chat.completions.create(
        model = "openrouter/owl-alpha",
        messages = [
            {"role":"system","content":system_prompt},
            {"role":"user","content":f"you are given a query :{query} and its nicheness:{niche}. generate alternative queries when searched can possibly answer the query itself"}
            ]
        )
    content = completions.choices[0].message.content
    
    return content

